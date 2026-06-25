#!/usr/bin/env python3
"""Unit tests for the Flask API endpoints of the DRIMAPS Simulator."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_map_types_endpoint(client):
    response = client.get("/api/maps/types")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Must contain "random" and "warehouse"
    ids = [item["id"] for item in data]
    assert "random" in ids
    assert "warehouse" in ids


def test_map_generate_endpoint(client):
    response = client.post("/api/maps/generate", json={
        "map_type": "random",
        "size": 16,
        "density": 0.1,
        "seed": 42
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "grid" in data
    assert data["height"] == 16
    assert data["width"] == 16
    assert "difficulty" in data
    assert "difficulty_score" in data["difficulty"]


def test_scenario_generate_endpoint(client):
    # First generate a grid
    grid = np.zeros((8, 8), dtype=int).tolist()
    response = client.post("/api/scenario/generate", json={
        "grid": grid,
        "num_agents": 4,
        "seed": 42
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "starts" in data
    assert "goals" in data
    assert data["num_agents"] == 4
    assert len(data["starts"]) == 4
    assert len(data["goals"]) == 4


def test_simulate_endpoint(client):
    # Simple 4x4 grid with 2 agents
    grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ]
    starts = [[0, 0], [3, 3]]
    goals = [[3, 3], [0, 0]]

    # 1. Test DRIMAPS
    response = client.post("/api/simulate", json={
        "grid": grid,
        "starts": starts,
        "goals": goals,
        "algorithm": "drimaps"
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "trajectory" in data
    assert "metrics" in data
    assert data["metrics"]["collision_free"] is True
    assert data["metrics"]["isr"] == 1.0

    # 2. Test PIBT baseline
    response = client.post("/api/simulate", json={
        "grid": grid,
        "starts": starts,
        "goals": goals,
        "algorithm": "pibt"
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "trajectory" in data
    assert "metrics" in data
    assert data["metrics"]["collision_free"] is True
