import json
import uuid
import time
from enum import IntEnum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict, Union, List, Literal

class ActionCode(IntEnum):
    HANDSHAKE = 0
    CREATE_GROUP = 1
    PROBE_ALL = 2
    PROBE_ENV = 3
    PROBE_COMPUTE = 4
    PROBE_NET_INTRA = 5   # L1: Local COLLECTIVE/P2P within one node
    PROBE_NET_INTER = 6   # L2: Cross COLLECTIVE/P2P within multi nodes
    BARRIER = 7
    HEARTBEAT = 8
    TERMINATE = 9

class StatusCode(IntEnum):
    SUCCESS = 0
    ERROR = 1
    PENDING = 2  # Initial state for Futures

class MessageType(IntEnum):
    REQUEST = 0
    RESPONSE = 1
    EVENT = 2
    

@dataclass
class JanusMessage:
    source_rank: int
    target: Union[int, List[int], Literal["ALL"]]
    action: ActionCode
    msg_type: MessageType = MessageType.REQUEST
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: StatusCode = StatusCode.PENDING
    payload: Dict[str, Any] = field(default_factory=dict)
    error_msg: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'JanusMessage':
        data = json.loads(json_str)
        data['action'] = ActionCode(data['action'])
        data['status'] = StatusCode(data['status'])
        data['msg_type'] = MessageType(data.get('msg_type', 0))
        return cls(**data)

class JanusFuture:
    """L3: Task Abstraction Layer."""
    def __init__(self, request_id: str, action: ActionCode, collective: 'JanusMasterCollective', expected_count: int):
        self.request_id = request_id
        self.action = action
        self.collective = collective
        self.expected_count = expected_count
        self._result = None

    def done(self) -> bool:
        """Non-blocking check if the expected number of responses has arrived."""
        with self.collective.messenger.inbox_lock:
            return len(self.collective.messenger.inbox.get(self.request_id, [])) >= self.expected_count

    def result(self, timeout: float = 60.0, min_success_ratio: float = 1.0) -> Dict[str, Any]:
        """Blocking wait for task completion with quorum semantics."""
        if self._result: 
            return self._result
            
        self._result = self.collective.gather_reports(
            self.action,
            self.request_id, 
            min_success_ratio=min_success_ratio, 
            timeout=timeout
        )
        return self._result