"""Atomic bounded concurrency and cost admission for gateway generations."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Literal, NoReturn

from headroom.proxy.gateway.errors import GatewayAuthorizationError


@dataclass(frozen=True, slots=True)
class AdmissionRequest:
    principal_id: str
    estimated_cost: float | None
    queue_timeout: float = 0.0


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    allowed: bool
    reason: str | None
    reservation: AdmissionReservation | None


class AdmissionReservation:
    def __init__(
        self,
        controller: AdmissionController,
        reservation_id: str,
        estimated_cost: float | None,
    ) -> None:
        self._controller = controller
        self._reservation_id = reservation_id
        self._estimated_cost = estimated_cost
        self._closed = False

    async def finalize(self, *, actual_cost: float | None) -> None:
        if self._closed:
            return
        self._closed = True
        await self._controller._close(
            self._reservation_id,
            estimated_cost=self._estimated_cost,
            actual_cost=actual_cost,
            committed=True,
        )

    async def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._controller._close(
            self._reservation_id,
            estimated_cost=self._estimated_cost,
            actual_cost=None,
            committed=False,
        )

    async def __aenter__(self) -> AdmissionReservation:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        if exc_type is None:
            await self.finalize(actual_cost=None)
        else:
            await self.release()


class AdmissionController:
    def __init__(
        self,
        *,
        budget_limit: float | None,
        max_concurrency: int,
        queue_limit: int,
        unknown_cost_policy: Literal["allow", "block"],
    ) -> None:
        if max_concurrency < 1 or queue_limit < 0:
            raise ValueError("gateway admission bounds are invalid")
        if budget_limit is not None and budget_limit < 0:
            raise ValueError("gateway budget limit is invalid")
        self._budget_limit = budget_limit
        self._max_concurrency = max_concurrency
        self._queue_limit = queue_limit
        self._unknown_cost_policy = unknown_cost_policy
        self._active: dict[str, float | None] = {}
        self._reserved_cost = 0.0
        self._committed_cost = 0.0
        self._waiting = 0
        self._condition = asyncio.Condition()

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def committed_cost(self) -> float:
        return self._committed_cost

    async def try_reserve(self, request: AdmissionRequest) -> AdmissionResult:
        async with self._condition:
            reason = self._denial_reason(request)
            if reason is not None:
                return AdmissionResult(False, reason, None)
            return AdmissionResult(True, None, self._admit(request))

    async def reserve(self, request: AdmissionRequest) -> AdmissionReservation:
        result = await self.try_reserve(request)
        if result.reservation is not None:
            return result.reservation
        if result.reason != "concurrency" or request.queue_timeout <= 0:
            self._raise_denied(result.reason)

        async with self._condition:
            if self._waiting >= self._queue_limit:
                self._raise_denied("queue_full")
            self._waiting += 1
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._denial_reason(request) != "concurrency"),
                    timeout=request.queue_timeout,
                )
                reason = self._denial_reason(request)
                if reason is not None:
                    self._raise_denied(reason)
                return self._admit(request)
            except asyncio.TimeoutError:
                self._raise_denied("queue_timeout")
            finally:
                self._waiting -= 1

    def _denial_reason(self, request: AdmissionRequest) -> str | None:
        if request.estimated_cost is None and self._unknown_cost_policy == "block":
            return "unknown_cost"
        if len(self._active) >= self._max_concurrency:
            return "concurrency"
        estimate = request.estimated_cost or 0.0
        if (
            self._budget_limit is not None
            and self._committed_cost + self._reserved_cost + estimate > self._budget_limit
        ):
            return "budget"
        return None

    def _admit(self, request: AdmissionRequest) -> AdmissionReservation:
        reservation_id = uuid.uuid4().hex
        self._active[reservation_id] = request.estimated_cost
        self._reserved_cost += request.estimated_cost or 0.0
        return AdmissionReservation(self, reservation_id, request.estimated_cost)

    async def _close(
        self,
        reservation_id: str,
        *,
        estimated_cost: float | None,
        actual_cost: float | None,
        committed: bool,
    ) -> None:
        async with self._condition:
            if reservation_id not in self._active:
                return
            self._active.pop(reservation_id)
            self._reserved_cost -= estimated_cost or 0.0
            if committed and actual_cost is not None:
                self._committed_cost += actual_cost
            self._condition.notify(1)

    @staticmethod
    def _raise_denied(reason: str | None) -> NoReturn:
        raise GatewayAuthorizationError(
            status_code=429 if reason in {"concurrency", "queue_full", "queue_timeout"} else 403,
            code=f"gateway_admission_{reason or 'denied'}",
            message=f"Gateway admission denied: {reason or 'unknown'}",
        )
