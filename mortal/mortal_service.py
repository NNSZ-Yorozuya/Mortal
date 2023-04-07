import gc
import os
import uuid
from datetime import datetime, timezone
from multiprocessing import Lock
from threading import Thread
from time import time, sleep
from typing import Optional

import torch
from flask import Flask, request, make_response
from mortal_service_data import get_events, get_sessions, insert_event, insert_session
from libriichi.mjai import Bot
from readerwriterlock.rwlock import RWLockRead

from config import config
from engine import MortalEngine
from model import Brain, DQN

app = Flask(__name__)

device = torch.device('cpu')
state = torch.load(config['control']['state_file'],
                   map_location=torch.device('cpu'))
cfg = state['config']
version = cfg['control'].get('version', 1)
num_blocks = cfg['resnet']['num_blocks']
conv_channels = cfg['resnet']['conv_channels']
timestamp = datetime.fromtimestamp(
    state['timestamp'], tz=timezone.utc).strftime('%y%m%d%H')
tag = f'mortal{version}-b{num_blocks}c{conv_channels}-t{timestamp}'


class Session:
    __slots__ = ("session_id", "player_id", "destroyed", "bot", "lock")

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.destroyed = False

        mortal = Brain(version=version, num_blocks=num_blocks,
                       conv_channels=conv_channels).eval()
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

        self.session_id = insert_session(player_id)

    def destroy(self):
        with self.lock:
            self.destroyed = True
            self.bot = None
            gc.collect()

    def react(self, event: str) -> Optional[str]:
        with self.lock:
            reaction = self.bot.react(event.strip())
            insert_event(self.session_id, event, reaction)
        return reaction


class SessionManager:
    __slots__ = ("sessions", "lock", "clean_deamon_thread")

    def __init__(self):
        self.sessions = {}
        self.lock = RWLockRead()

        def clean_worker():
            while True:
                try:
                    with self.lock.gen_wlock():
                        now = time()
                        for o in self.sessions.values():
                            if now - o['last_access_time'] > 300:
                                app.logger.info(
                                    f'remove inactive session {o["session"].session_id}')
                except Exception as e:
                    app.logger.exception(e)

                sleep(60)

        self.clean_deamon_thread = Thread(
            target=clean_worker, name="session_manager_clean_deamon")
        self.clean_deamon_thread.start()

    def get(self, session_id: int) -> Optional[Session]:
        with self.lock.gen_rlock():
            o = self.sessions.get(session_id, None)
            o['last_access_time'] = time()
            return o['session']

    def new(self, **kwargs) -> Session:
        session = Session(**kwargs)

        with self.lock.gen_wlock():
            self.sessions[session.session_id] = {
                "session": session,
                "last_access_time": time()
            }

        return session

    def destroy(self, session_id: int):
        with self.lock.gen_wlock():
            o = self.sessions.pop(session_id)
        o['session'].destroy()


mgr = SessionManager()


@app.post("/sessions")
def _post_session():
    player_id = request.args.get('player_id', type=int)
    s = mgr.new(player_id=player_id)
    return {"session_id": s.session_id}


@app.get("/sessions")
def _get_sessions():
    sessions = get_sessions()
    return sessions


@app.post("/sessions/<int:session_id>")
def _post_session_event(session_id: int):
    s = mgr.get(session_id)

    line = request.data.decode(encoding='utf-8')
    app.logger.debug(f"< {line}")

    reaction = s.react(line)
    app.logger.debug(f"> {reaction}")

    if reaction is None:
        reaction = ''

    resp = make_response(reaction, 200)
    resp.mimetype = "application/json"
    return resp


@app.get("/sessions/<int:session_id>")
def _get_session_events(session_id: int):
    events = get_events(session_id)

    content = '[' + ', '.join([e['content'] for e in events]) + ']'
    resp = make_response(content, 200)
    resp.mimetype = "application/json"
    return resp


@app.route("/")
def _hello():
    return "Hello, World!"


if __name__ == '__main__':
    host = os.getenv("MORTAL_HOST") or '127.0.0.1'
    port = os.getenv("MORTAL_PORT") or 5000
    app.run(host=host, port=port)
