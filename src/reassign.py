"""Work out who should cover what, so nobody breaches.

This is a constrained assignment problem, not a ranking. Moving a shift off an
at-risk person has to land it on somebody who can legally take it: same site,
same role, same day/night pattern, not already working that day, and with enough
room left in their own week that we do not simply move the breach along. Receiver
capacity makes it a knapsack per person, which is what stops a greedy matcher
from working - it fixes the first name on the list and creates a new problem
three rows down.

Answer Set Programming suits this. The constraints read close to the rules they
encode, the solver searches exhaustively rather than greedily, and it optimises
over the whole plan instead of one person at a time. A greedy fallback is kept
for environments without clingo, and it is honest about being worse.
"""

from __future__ import annotations

from dataclasses import dataclass

from .roster import DAY_NAMES, ProbableShift

#: Hours are always quarter-hours, so scaling by 4 keeps the solver in integers.
SCALE = 4

#: Ignore probable shifts too unlikely to be worth planning around.
MIN_PROBABILITY = 0.35


@dataclass(frozen=True)
class Candidate:
    """Someone who could take extra work without getting into trouble themselves."""

    employee_id: str
    room_hours: float          # hours before they reach 45
    site_id: str
    role: str
    pattern: str


@dataclass(frozen=True)
class Move:
    """One suggested reassignment."""

    shift_ref: str
    from_employee: str
    to_employee: str
    weekday: int
    hours: float
    to_name: str = ""

    @property
    def day_name(self) -> str:
        return DAY_NAMES[self.weekday]


@dataclass
class Plan:
    moves: list[Move]
    unresolved: list[str]
    solver: str
    hours_moved: float = 0.0

    def by_employee(self) -> dict[str, list[Move]]:
        out: dict[str, list[Move]] = {}
        for move in self.moves:
            out.setdefault(move.from_employee, []).append(move)
        return out


def _quantise(hours: float) -> int:
    return int(round(hours * SCALE))


def build_program(
    at_risk: dict[str, float],
    shifts: list[ProbableShift],
    candidates: list[Candidate],
    busy: set[tuple[str, int]],
    profile: dict[str, tuple[str, str, str]],
) -> str:
    """Emit the logic program: facts, then the rules that constrain a valid plan."""
    lines: list[str] = ["% ---- facts ----"]

    for employee, shed in at_risk.items():
        # Priority weight: shedding more hours matters more.
        lines.append(f"at_risk({employee.lower()},{_quantise(shed)}).")

    movable = [s for s in shifts
               if s.employee_id in at_risk and s.probability >= MIN_PROBABILITY]
    for s in movable:
        site, role, pattern = profile[s.employee_id]
        lines.append(
            f"shift({s.shift_ref.lower()},{s.employee_id.lower()},{s.weekday},"
            f"{_quantise(s.hours)},{site.lower().replace('-','_')},"
            f"{role.lower().replace(' ','_')},{pattern.lower()})."
        )

    for c in candidates:
        lines.append(
            f"cover({c.employee_id.lower()},{_quantise(c.room_hours)},"
            f"{c.site_id.lower().replace('-','_')},{c.role.lower().replace(' ','_')},"
            f"{c.pattern.lower()})."
        )

    known = {c.employee_id for c in candidates}
    for employee, weekday in busy:
        if employee in known:
            lines.append(f"busy({employee.lower()},{weekday}).")

    lines += [
        "",
        "% ---- a shift may be handed to at most one person who can legally take it ----",
        "{ move(S,R) : cover(R,_,Site,Role,Pattern), not busy(R,Day) } 1"
        " :- shift(S,_,Day,_,Site,Role,Pattern).",
        "",
        "% nobody may be given more than their remaining room below 45 hours",
        ":- cover(R,Room,_,_,_), #sum{ H,S : move(S,R), shift(S,_,_,H,_,_,_) } > Room.",
        "",
        "% nobody may be handed two shifts on the same day",
        ":- move(S1,R), move(S2,R), S1 < S2,"
        " shift(S1,_,D,_,_,_,_), shift(S2,_,D,_,_,_,_).",
        "",
        "% an at-risk person is resolved once enough hours have been taken off them",
        "shed(E,H) :- at_risk(E,_), H = #sum{ Hs,S : move(S,_), shift(S,E,_,Hs,_,_,_) }.",
        "resolved(E) :- at_risk(E,Need), shed(E,H), H >= Need.",
        "",
        "% Resolve as many people as possible, weighted by how far over they are,",
        "% and prefer plans that disturb fewer shifts. Two levels rather than three:",
        "% a third tie-breaker roughly tripled the time to a good plan and never",
        "% changed which people ended up covered.",
        "#maximize { N@2,E : resolved(E), at_risk(E,N) }.",
        "#minimize { 1@1,S,R : move(S,R) }.",
        "",
        "#show move/2.",
        "#show resolved/1.",
    ]
    return "\n".join(lines)


def solve_with_clingo(program: str, at_risk: dict[str, float],
                      shifts: list[ProbableShift], time_limit: float = 15.0) -> Plan | None:
    """Search for the best plan within a time budget.

    Proving optimality on this program does not terminate in any useful time, and
    it does not need to: clingo emits a stream of successively better plans, so we
    keep the last one it managed before the budget ran out. Each improvement is
    captured in the callback - reading the handle after cancelling it yields
    nothing, which is the trap this replaced.
    """
    try:
        import clingo
    except ImportError:
        return None

    best: list[str] = []

    def capture(model) -> None:
        best.clear()
        best.extend(str(atom) for atom in model.symbols(shown=True))

    control = clingo.Control(["--models=0", "--opt-strategy=usc"])
    control.add("base", [], program)
    control.ground([("base", [])])

    with control.solve(on_model=capture, async_=True) as handle:
        handle.wait(time_limit)
        handle.cancel()

    if not best:
        return None

    lookup = {s.shift_ref.lower(): s for s in shifts}
    owner = {e.lower(): e for e in at_risk}
    moves, resolved = [], set()
    for atom in best:
        if atom.startswith("move("):
            ref, receiver = atom[5:-1].split(",")
            shift = lookup.get(ref)
            if shift is None:
                continue
            moves.append(Move(shift_ref=shift.shift_ref, from_employee=shift.employee_id,
                              to_employee=receiver.upper(), weekday=shift.weekday,
                              hours=shift.hours))
        elif atom.startswith("resolved("):
            resolved.add(owner.get(atom[9:-1], atom[9:-1].upper()))

    return Plan(moves=moves, solver="clingo",
                unresolved=sorted(set(at_risk) - resolved),
                hours_moved=sum(m.hours for m in moves))


def solve_greedy(at_risk: dict[str, float], shifts: list[ProbableShift],
                 candidates: list[Candidate], busy: set[tuple[str, int]],
                 profile: dict[str, tuple[str, str, str]]) -> Plan:
    """Fallback when clingo is unavailable. Handles each person in turn, so it can
    strand later names by spending capacity early - which is exactly the weakness
    the solver exists to avoid."""
    room = {c.employee_id: c.room_hours for c in candidates}
    taken = set(busy)
    by_employee: dict[str, list[ProbableShift]] = {}
    for s in shifts:
        if s.employee_id in at_risk and s.probability >= MIN_PROBABILITY:
            by_employee.setdefault(s.employee_id, []).append(s)

    moves, resolved = [], set()
    for employee, need in sorted(at_risk.items(), key=lambda kv: -kv[1]):
        site, role, pattern = profile[employee]
        shed = 0.0
        for shift in sorted(by_employee.get(employee, []), key=lambda s: -s.hours):
            if shed >= need:
                break
            options = [c for c in candidates
                       if c.site_id == site and c.role == role and c.pattern == pattern
                       and room.get(c.employee_id, 0) >= shift.hours
                       and (c.employee_id, shift.weekday) not in taken]
            if not options:
                continue
            best = max(options, key=lambda c: room[c.employee_id])
            room[best.employee_id] -= shift.hours
            taken.add((best.employee_id, shift.weekday))
            moves.append(Move(shift.shift_ref, employee, best.employee_id,
                              shift.weekday, shift.hours))
            shed += shift.hours
        if shed >= need:
            resolved.add(employee)

    return Plan(moves=moves, solver="greedy",
                unresolved=sorted(set(at_risk) - resolved),
                hours_moved=sum(m.hours for m in moves))
