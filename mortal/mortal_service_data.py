from typing import Optional
from sqlalchemy import Column, StaticPool, Table, Integer, DateTime, MetaData, Text, create_engine, func, insert, select


meta = MetaData()

session_table = Table(
    "session",
    meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("player_id", Integer),
    Column("created_at", DateTime, server_default=func.now())
)

event_table = Table(
    "event",
    meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", Integer, index=True),
    Column("content", Text),
    Column("reaction", Text, nullable=True)
)


engine = create_engine("sqlite:///mortal_service.db", poolclass=StaticPool)

meta.create_all(engine)


def insert_session(player_id: int) -> int:
    stmt = insert(session_table).values(player_id=player_id)
    with engine.connect() as conn:
        result = conn.execute(stmt)
        conn.commit()

        return result.inserted_primary_key[0]


def get_sessions():
    sessions = []

    stmt = select(session_table).order_by(session_table.c.id)
    with engine.connect() as conn:
        result = conn.execute(stmt)
        for row in result:
            sessions.append({
                "session_id": row.id,
                "player_id": row.player_id,
                "created_at": row.created_at
            })

    return sessions


def insert_event(session_id: int, content: str, reaction: Optional[str]) -> int:
    stmt = (insert(event_table)
            .values(session_id=session_id,
                    content=content, reaction=reaction))
    with engine.connect() as conn:
        result = conn.execute(stmt)
        conn.commit()

        return result.inserted_primary_key[0]


def get_events(session_id: int):
    events = []

    stmt = (select(event_table)
            .where(event_table.c.session_id == session_id)
            .order_by(event_table.c.id))
    with engine.connect() as conn:
        result = conn.execute(stmt)
        for row in result:
            events.append({
                "content": row.content,
                "reaction": row.reaction
            })

    return events


__all__ = ("insert_session", "get_sessions", "insert_event", "get_events")
