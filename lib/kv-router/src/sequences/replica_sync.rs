// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::future::poll_fn;
use std::sync::Arc;
use std::task::Poll;

use rustc_hash::FxHashMap;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use super::multi_worker::{
    ActiveSequencesMultiWorker, ReplicaWorkerPolicy, SequencePublisher, SequenceSubscriber,
};
use super::prompt_registry::WorkerLoadSnapshot;
use crate::protocols::{ActiveSequenceEvent, ActiveSequenceEventData, WorkerWithDpRank};

const MAX_REPLICA_BATCH_EVENTS: usize = 256;
const MAX_REPLICA_BATCH_DURATION: Duration = Duration::from_millis(1);
const REPLICA_REORDER_STATE_TTL: Duration = Duration::from_secs(300);

type ReplicaLifecycleKey = (u64, String);

/// Short-lived state for lifecycle events that overtake AddRequest in the transport.
///
/// The publisher serializes new events per request, but this also protects rolling upgrades
/// and transports with at-least-once delivery or multiple in-flight publishers.
#[derive(Default)]
struct ReplicaLifecycleReorderState {
    free_tombstones: HashMap<ReplicaLifecycleKey, Instant>,
    free_tombstone_order: VecDeque<(Instant, ReplicaLifecycleKey)>,
    pending_prefill_completed: HashMap<ReplicaLifecycleKey, Instant>,
    pending_prefill_completed_order: VecDeque<(Instant, ReplicaLifecycleKey)>,
}

impl ReplicaLifecycleReorderState {
    fn prune(&mut self, now: Instant) {
        Self::prune_entries(
            &mut self.free_tombstones,
            &mut self.free_tombstone_order,
            now,
        );
        Self::prune_entries(
            &mut self.pending_prefill_completed,
            &mut self.pending_prefill_completed_order,
            now,
        );
    }

    fn prune_entries(
        entries: &mut HashMap<ReplicaLifecycleKey, Instant>,
        order: &mut VecDeque<(Instant, ReplicaLifecycleKey)>,
        now: Instant,
    ) {
        while order.front().is_some_and(|(observed_at, _)| {
            now.saturating_duration_since(*observed_at) > REPLICA_REORDER_STATE_TTL
        }) {
            let (observed_at, key) = order
                .pop_front()
                .expect("front entry must exist while pruning replica lifecycle state");
            if entries.get(&key) == Some(&observed_at) {
                entries.remove(&key);
            }
        }
    }

    fn record_free(&mut self, key: ReplicaLifecycleKey, now: Instant) {
        self.free_tombstones.insert(key.clone(), now);
        self.free_tombstone_order.push_back((now, key));
    }

    fn was_freed(&self, key: &ReplicaLifecycleKey) -> bool {
        self.free_tombstones.contains_key(key)
    }

    fn record_pending_prefill_completed(&mut self, key: ReplicaLifecycleKey, now: Instant) {
        self.pending_prefill_completed.insert(key.clone(), now);
        self.pending_prefill_completed_order.push_back((now, key));
    }

    fn take_pending_prefill_completed(&mut self, key: &ReplicaLifecycleKey) -> bool {
        self.pending_prefill_completed.remove(key).is_some()
    }
}

#[derive(Default)]
struct ReplicaBatchEffects {
    worker_loads: FxHashMap<WorkerWithDpRank, PendingLoadPublication>,
    wake_scheduler: bool,
    cleanup_prompt_trie: bool,
}

struct PendingLoadPublication {
    latest_load: WorkerLoadSnapshot,
    publish: bool,
}

impl ReplicaBatchEffects {
    fn record_worker_load(
        &mut self,
        worker: WorkerWithDpRank,
        load: WorkerLoadSnapshot,
        publish: bool,
    ) {
        self.worker_loads
            .entry(worker)
            .and_modify(|pending| {
                pending.latest_load = load;
                pending.publish |= publish;
            })
            .or_insert(PendingLoadPublication {
                latest_load: load,
                publish,
            });
    }
}

impl<P: SequencePublisher + 'static> ActiveSequencesMultiWorker<P> {
    /// Spawn a background task that subscribes to replica-sync events from peer routers
    /// and applies them to the local state.
    pub fn start_replica_sync<S: SequenceSubscriber + 'static>(
        self: &Arc<Self>,
        subscriber: S,
        cancel_token: CancellationToken,
    ) {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = this.run_replica_sync(subscriber, cancel_token).await {
                tracing::error!("Error in active sequences events subscription: {error}");
            }
        });
    }

    pub(super) async fn run_replica_sync<S: SequenceSubscriber>(
        &self,
        mut subscriber: S,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        let mut effects = ReplicaBatchEffects::default();
        let mut reorder_state = ReplicaLifecycleReorderState::default();

        loop {
            let result = tokio::select! {
                result = subscriber.next_event() => result,
                _ = cancel_token.cancelled() => {
                    tracing::debug!("Subscription task cancelled");
                    break;
                }
            };

            let Some(result) = result else {
                break;
            };
            let Ok(event) = result else {
                tracing::error!(
                    "Error receiving active sequence event: {}",
                    result.unwrap_err()
                );
                continue;
            };

            let batch_start = Instant::now();
            let mut batch_events = 0;
            let mut exit_after_flush = false;
            let mut yield_after_flush = false;
            let mut next_event = Some(event);

            loop {
                let event = next_event
                    .take()
                    .expect("replica batch event must be present");
                self.apply_replica_event(event, &mut effects, &mut reorder_state);
                batch_events += 1;

                if cancel_token.is_cancelled() {
                    exit_after_flush = true;
                    break;
                }

                if batch_events >= MAX_REPLICA_BATCH_EVENTS
                    || batch_start.elapsed() >= MAX_REPLICA_BATCH_DURATION
                {
                    yield_after_flush = true;
                    break;
                }

                match poll_fn(|cx| Poll::Ready(subscriber.poll_next_event(cx))).await {
                    Poll::Ready(Some(Ok(event))) => next_event = Some(event),
                    Poll::Ready(Some(Err(error))) => {
                        tracing::error!("Error receiving active sequence event: {error}");
                        break;
                    }
                    Poll::Ready(None) => {
                        exit_after_flush = true;
                        break;
                    }
                    Poll::Pending => break,
                }
            }

            self.flush_replica_batch_effects(&mut effects);

            if exit_after_flush {
                break;
            }
            if yield_after_flush {
                tokio::task::yield_now().await;
            }
        }

        Ok(())
    }

    fn apply_replica_event(
        &self,
        event: ActiveSequenceEvent,
        effects: &mut ReplicaBatchEffects,
        reorder_state: &mut ReplicaLifecycleReorderState,
    ) {
        let ActiveSequenceEvent {
            request_id,
            worker: event_worker,
            data,
            router_id,
            lora_name,
        } = event;

        if router_id == self.router_id {
            return;
        }

        // ActiveSequenceEvent does not carry prompt-load decay timestamps yet.
        // Peer routers still approximate decay anchoring with local receive time.
        let decay_now = Instant::now();
        reorder_state.prune(decay_now);
        let lifecycle_key = (router_id, request_id.clone());

        match data {
            ActiveSequenceEventData::AddRequest {
                token_sequence,
                track_prefill_tokens,
                expected_output_tokens,
                prefill_load_hint,
            } => {
                if reorder_state.was_freed(&lifecycle_key) {
                    reorder_state.take_pending_prefill_completed(&lifecycle_key);
                    tracing::debug!(
                        request_id = %request_id,
                        worker = ?event_worker,
                        router_id,
                        "Dropping replica AddRequest superseded by an earlier Free"
                    );
                    return;
                }
                let prefill_already_completed =
                    reorder_state.take_pending_prefill_completed(&lifecycle_key);
                if self.replica_worker_policy == ReplicaWorkerPolicy::LazyRegister {
                    self.ensure_worker_registered(event_worker);
                }
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&event_worker) else {
                    tracing::debug!(
                        worker = ?event_worker,
                        "Dropping replica AddRequest for unregistered worker"
                    );
                    return;
                };

                self.request_index
                    .set_request(request_id.clone(), event_worker, lora_name);
                let (expired_request_ids, load) = {
                    let slot = &table.slots[idx];
                    let mut seq = slot.sequences.write();
                    let prefill_completed_request_id =
                        prefill_already_completed.then(|| request_id.clone());
                    let outcome = seq.add_request_with_prefill_tracking(
                        request_id,
                        token_sequence,
                        expected_output_tokens,
                        track_prefill_tokens,
                        prefill_load_hint,
                        decay_now,
                    );
                    if let Some(request_id) = prefill_completed_request_id {
                        seq.mark_prefill_completed(&request_id, decay_now);
                    }
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry
                        .apply_membership_delta_and_load_without_cleanup(
                            event_worker,
                            &slot.trie_lookup,
                            outcome.membership_delta,
                            load,
                        );
                    (outcome.expired_request_ids, load)
                };
                drop(table);
                self.request_index
                    .remove_requests(expired_request_ids.iter());
                effects.record_worker_load(event_worker, load, true);
                effects.wake_scheduler |= prefill_already_completed;
                effects.cleanup_prompt_trie = true;
            }
            ActiveSequenceEventData::Free => {
                let Some(worker) = self.request_index.remove_request(&request_id) else {
                    reorder_state.take_pending_prefill_completed(&lifecycle_key);
                    reorder_state.record_free(lifecycle_key, decay_now);
                    return;
                };
                reorder_state.take_pending_prefill_completed(&lifecycle_key);
                reorder_state.record_free(lifecycle_key, decay_now);
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&worker) else {
                    return;
                };
                let load = {
                    let slot = &table.slots[idx];
                    let mut seq = slot.sequences.write();
                    let delta = seq.free(&request_id, decay_now);
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry
                        .apply_membership_delta_and_load_without_cleanup(
                            worker,
                            &slot.trie_lookup,
                            delta,
                            load,
                        );
                    load
                };
                drop(table);
                effects.record_worker_load(worker, load, true);
                effects.wake_scheduler = true;
                effects.cleanup_prompt_trie = true;
            }
            ActiveSequenceEventData::MarkPrefillCompleted => {
                let Some(worker) = self.request_index.worker_for(&request_id) else {
                    reorder_state.record_pending_prefill_completed(lifecycle_key, decay_now);
                    return;
                };
                reorder_state.take_pending_prefill_completed(&lifecycle_key);
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&worker) else {
                    return;
                };
                let load = {
                    let mut seq = table.slots[idx].sequences.write();
                    seq.mark_prefill_completed(&request_id, decay_now);
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry.replace_worker_load_state(worker, load);
                    load
                };
                drop(table);
                effects.record_worker_load(worker, load, false);
                effects.wake_scheduler = true;
            }
        }
    }

    fn flush_replica_batch_effects(&self, effects: &mut ReplicaBatchEffects) {
        let decay_now = Instant::now();
        let mut active_loads = Vec::with_capacity(effects.worker_loads.len());
        for (worker, pending) in effects.worker_loads.drain() {
            if pending.publish {
                active_loads.push(self.observe_worker_load_snapshot(
                    worker,
                    pending.latest_load,
                    decay_now,
                ));
            }
        }
        if !active_loads.is_empty() {
            self.publisher.publish_load_batch(active_loads);
        }

        if std::mem::take(&mut effects.wake_scheduler) {
            self.notify_remote_state_update();
        }
        if std::mem::take(&mut effects.cleanup_prompt_trie) {
            self.prompt_registry.maybe_cleanup();
        }
    }
}
