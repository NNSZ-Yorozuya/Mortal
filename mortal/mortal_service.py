import gc
import uuid
from datetime import datetime, timezone
from multiprocessing import Lock
from time import time
from typing import Optional

import torch
from flask import Flask, request, make_response
from libriichi.mjai import Bot
from readerwriterlock.rwlock import RWLockRead

from config import config
from engine import MortalEngine
from model import Brain, DQN

app = Flask(__name__)

device = torch.device('cpu')
state = torch.load(config['control']['state_file'], map_location=torch.device('cpu'))
cfg = state['config']
version = cfg['control'].get('version', 1)
num_blocks = cfg['resnet']['num_blocks']
conv_channels = cfg['resnet']['conv_channels']
timestamp = datetime.fromtimestamp(state['timestamp'], tz=timezone.utc).strftime('%y%m%d%H')
tag = f'mortal{version}-b{num_blocks}c{conv_channels}-t{timestamp}'


class Session:
    __slots__ = ("session_id", "player_id", "destroyed", "bot", "lock")

    def __init__(self, session_id: str, player_id: int):
        self.session_id = session_id
        self.player_id = player_id
        self.destroyed = False

        mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).eval()
        dqn = DQN(version=version).eval()
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])

        engine = MortalEngine(
            mortal,
            dqn,
            version=version,
            is_oracle=False,
            device=device,
            enable_amp=False,
            enable_quick_eval=True,
            enable_rule_based_agari_guard=True,
            name='mortal',
        )

        self.bot = Bot(engine, player_id)

        self.lock = Lock()

    def destroy(self):
        with self.lock:
            self.destroyed = True
            self.bot = None
            gc.collect()

    def react(self, line: str) -> Optional[str]:
        t = time()
        with self.lock:
            t2 = time()
            r = self.bot.react(line.strip())
        t3 = time()
        app.logger.debug(f"cost: {'%.2f' % (t3 - t)}s  "
                         f"(waiting: {'%.2f' % (t2 - t)}s, "
                         f"calculation: {'%.2f' % (t3 - t2)}s)")
        return r


class SessionManager:
    __slots__ = ("sessions", "lock")

    def __init__(self):
        self.sessions = {}
        self.lock = RWLockRead()

    def get(self, session_id: str) -> Optional[Session]:
        with self.lock.gen_rlock():
            return self.sessions.get(session_id, None)

    def new(self, **kwargs) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id, **kwargs)

        with self.lock.gen_wlock():
            self.sessions[session_id] = session

        return session

    def destroy(self, session_id: str):
        with self.lock.gen_wlock():
            session = self.sessions.pop(session_id)
        session.destroy()


mgr = SessionManager()


@app.post("/session")
def post_session():
    player_id = request.args.get('player_id', type=int)
    s = mgr.new(player_id=player_id)
    return {"session_id": s.session_id}


@app.post("/session/<session_id>")
def post_session_react(session_id: str):
    s = mgr.get(session_id)

    line = request.data.decode(encoding='utf-8')
    app.logger.debug(f"< {line}")

    react = s.react(line)
    app.logger.debug(f"> {react}")

    if react is None:
        react = ''

    resp = make_response(react, 200)
    resp.mimetype = "application/json"
    return resp


@app.delete("/session/<session_id>")
def delete_session(session_id: str):
    mgr.destroy(session_id)
    return ''
