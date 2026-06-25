#!/usr/bin/env python3
"""Unit tests for the cycle detector."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dependency_graph import WaitForGraph
from src.cycle_detector import CycleDetector


class TestCycleDetector:

    def test_no_cycles(self):
        wfg = WaitForGraph(3)
        wfg.add_edge(0, 1)
        wfg.add_edge(1, 2)
        detector = CycleDetector()
        cycles = detector.detect_cycles(wfg)
        assert len(cycles) == 0

    def test_simple_cycle(self):
        wfg = WaitForGraph(2)
        wfg.add_edge(0, 1)
        wfg.add_edge(1, 0)
        detector = CycleDetector()
        cycles = detector.detect_cycles(wfg)
        assert len(cycles) == 1
        assert set(cycles[0]) == {0, 1}

    def test_three_agent_cycle(self):
        wfg = WaitForGraph(3)
        wfg.add_edge(0, 1)
        wfg.add_edge(1, 2)
        wfg.add_edge(2, 0)
        detector = CycleDetector()
        cycles = detector.detect_cycles(wfg)
        assert len(cycles) == 1
        assert set(cycles[0]) == {0, 1, 2}

    def test_multiple_cycles(self):
        wfg = WaitForGraph(4)
        wfg.add_edge(0, 1)
        wfg.add_edge(1, 0)
        wfg.add_edge(2, 3)
        wfg.add_edge(3, 2)
        detector = CycleDetector()
        cycles = detector.detect_cycles(wfg)
        assert len(cycles) == 2

    def test_finished_agents_excluded(self):
        wfg = WaitForGraph(3)
        wfg.add_edge(0, 1)
        wfg.add_edge(1, 0)
        detector = CycleDetector()
        cycles = detector.detect_cycles(wfg, finished={0})
        # Agent 0 is finished, so no cycle
        assert len(cycles) == 0

    def test_has_deadlock(self):
        wfg = WaitForGraph(2)
        wfg.add_edge(0, 1)
        wfg.add_edge(1, 0)
        detector = CycleDetector()
        assert detector.has_deadlock(wfg)

    def test_no_deadlock(self):
        wfg = WaitForGraph(3)
        wfg.add_edge(0, 1)
        detector = CycleDetector()
        assert not detector.has_deadlock(wfg)

    def test_self_loop_not_cycle(self):
        """A self-loop is not a deadlock (SCC of size 1)."""
        wfg = WaitForGraph(2)
        wfg.add_edge(0, 0)  # Self-loop
        detector = CycleDetector()
        cycles = detector.detect_cycles(wfg)
        # Self-loop creates an SCC of size 1 which is filtered out
        assert len(cycles) == 0

    def test_large_cycle(self):
        n = 10
        wfg = WaitForGraph(n)
        for i in range(n):
            wfg.add_edge(i, (i + 1) % n)
        detector = CycleDetector()
        cycles = detector.detect_cycles(wfg)
        assert len(cycles) == 1
        assert len(cycles[0]) == n
