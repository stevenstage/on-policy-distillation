"""
Multi-hop Knowledge-Graph QA Environment
=========================================
Graph:  Person --[works_at]--> Company --[located_in]--> Country

Task:   Given person P, find which country they live in.
Optimal trajectory: lookup_person(P) -> lookup_company(C) -> answer(N)  [3 steps]

Action space  (n_actions = N_P + N_C + N_N):
    [0,      N_P)         lookup_person(i)
    [N_P,    N_P+N_C)     lookup_company(j)
    [N_P+N_C, ...)        answer(k)

Observation (obs_dim = 2*N_P + 2*N_C + N_N):
    [0,          N_P)     one-hot: question person
    [N_P,       2N_P)     binary:  which persons have been looked up
    [2N_P,  2N_P+N_C)     binary:  which companies have been looked up
    [2N_P+N_C, 2N_P+2N_C) one-hot: company of question-person (if known)
    [2N_P+2N_C, ...)      one-hot: country of question-company (if known)

Properties satisfying the assignment:
  ✓ Multi-turn decision (optimal: 3 steps, max: 8 steps)
  ✓ Tool-based actions  (lookup_person, lookup_company, answer)
  ✓ Error branches      (looking up the wrong entities wastes budget)
  ✓ Final-answer reward (+1 correct, 0 wrong/timeout)
  ✓ Early errors affect later state  (wrong company → wrong country)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class KGQAEnv:
    def __init__(
        self,
        n_persons:   int = 20,
        n_companies: int = 20,
        n_countries: int = 10,
        max_steps:   int = 8,
        seed:        int = 42,
    ):
        self.N_P = n_persons
        self.N_C = n_companies
        self.N_N = n_countries
        self.max_steps = max_steps

        # Per-episode RNG.  The knowledge graph is RE-RANDOMISED on every
        # reset(), so the answer can NOT be memorised as a fixed function of
        # the question-person one-hot.  The agent is forced to discover the
        # current episode's edges through the lookup chain — this is what makes
        # the task genuinely multi-hop (early lookup errors change which edges
        # ever get revealed, propagating to the final answer).
        self._rng = np.random.RandomState(seed)
        # Placeholder graph; overwritten on first reset().
        self.person_company  = self._rng.randint(0, n_companies, n_persons)
        self.company_country = self._rng.randint(0, n_countries, n_companies)

        self.n_actions = n_persons + n_companies + n_countries   # 50
        self.obs_dim   = 2*n_persons + 2*n_companies + n_countries  # 90

    # ── public helpers ────────────────────────────────────────────────────────

    def correct_answer(self, person_id: int) -> int:
        return int(self.company_country[self.person_company[person_id]])

    def oracle_action(self) -> int:
        """Optimal (oracle) action given current state."""
        if self.q not in self.seen_P:
            return self.q                                     # lookup_person(q)
        qc = self.seen_P[self.q]
        if qc not in self.seen_C:
            return self.N_P + qc                             # lookup_company(qc)
        return self.N_P + self.N_C + self.seen_C[qc]        # answer(country)

    def state_key(self) -> tuple:
        """Compact hashable key for off-support detection."""
        return (
            self.q,
            frozenset(self.seen_P.keys()),
            frozenset(self.seen_C.keys()),
        )

    # ── core interface ────────────────────────────────────────────────────────

    def reset(self, person_id: Optional[int] = None,
              graph: Optional[tuple] = None) -> np.ndarray:
        # Re-randomise the graph every episode so person→country is not a
        # fixed memorisable mapping.  The agent must execute lookup_person →
        # lookup_company to reveal the edges for THIS episode before it can
        # answer correctly.
        #
        # `graph` lets callers RESTORE a specific episode's graph so that a
        # trajectory can be faithfully replayed (needed to recompute oracle
        # teacher labels offline).  Pass the tuple returned by get_graph().
        if graph is not None:
            self.person_company, self.company_country = (
                np.array(graph[0]), np.array(graph[1]))
        else:
            self.person_company  = self._rng.randint(0, self.N_C, self.N_P)
            self.company_country = self._rng.randint(0, self.N_N, self.N_C)

        self.q      = int(person_id if person_id is not None
                          else self._rng.randint(self.N_P))
        self.target = self.correct_answer(self.q)
        self.seen_P: Dict[int, int] = {}   # person  -> company
        self.seen_C: Dict[int, int] = {}   # company -> country
        self.steps  = 0
        return self._obs()

    def get_graph(self) -> tuple:
        """Snapshot the current episode's graph for faithful replay."""
        return (self.person_company.copy(), self.company_country.copy())

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        info: dict = {"type": "", "step": self.steps}

        if action < self.N_P:                      # lookup_person
            pid = action
            cid = int(self.person_company[pid])
            self.seen_P[pid] = cid
            info.update(type="lookup_person", person=pid, company=cid)
            reward, done = 0.0, self.steps >= self.max_steps

        elif action < self.N_P + self.N_C:         # lookup_company
            cid = action - self.N_P
            nid = int(self.company_country[cid])
            self.seen_C[cid] = nid
            info.update(type="lookup_company", company=cid, country=nid)
            reward, done = 0.0, self.steps >= self.max_steps

        else:                                       # answer
            pred    = action - self.N_P - self.N_C
            correct = pred == self.target
            reward  = 1.0 if correct else 0.0
            done    = True
            info.update(type="answer", pred=pred, target=self.target,
                        correct=correct)

        return self._obs(), reward, done, info

    # ── private ───────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        o = np.zeros(self.obs_dim, dtype=np.float32)
        o[self.q] = 1.0                                            # question
        for p in self.seen_P:   o[self.N_P + p] = 1.0            # seen persons
        for c in self.seen_C:   o[2*self.N_P + c] = 1.0          # seen companies
        if self.q in self.seen_P:                                  # known company
            o[2*self.N_P + self.N_C + self.seen_P[self.q]] = 1.0
        qc = self.seen_P.get(self.q)                              # known country
        if qc is not None and qc in self.seen_C:
            o[2*self.N_P + 2*self.N_C + self.seen_C[qc]] = 1.0
        return o