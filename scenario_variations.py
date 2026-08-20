"""Reusable Phase 4 evacuation-scenario generator.

The notebook calls ``build_evacuation_scenarios`` after the baseline plan and
safe-gate routing logic have been created.  Every scenario is derived from a
deep copy of the baseline, so the baseline files and data frames remain the
reference case.
"""

from __future__ import annotations

from hashlib import blake2b
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_GENERATOR_VERSION = "phase4_scenarios_v1"


def _stable_hash(value, salt):
    """Return a deterministic unsigned integer without using random state."""
    payload = f"{salt}|{value}".encode("utf-8")
    return int(blake2b(payload, digest_size=8).hexdigest(), 16)


def _largest_remainder_quotas(weights, total, tie_breakers):
    """Allocate an integer total proportionally, preserving it exactly."""
    weights = pd.Series(weights, dtype="float64").fillna(0.0).clip(lower=0.0)
    total = int(total)
    if total < 0 or total > int(weights.sum()):
        raise ValueError("Scenario quota is outside the available population.")
    if total == 0:
        return pd.Series(0, index=weights.index, dtype="int64")
    if weights.sum() <= 0:
        raise ValueError("A positive quota needs positive candidate weights.")

    raw = weights / weights.sum() * total
    quota = np.floor(raw).astype("int64")
    remaining = total - int(quota.sum())
    if remaining:
        ties = pd.Series(tie_breakers, index=weights.index).astype(str)
        order = pd.DataFrame(
            {"fraction": raw - quota, "tie": ties}, index=weights.index
        ).sort_values(["fraction", "tie"], ascending=[False, True], kind="stable")
        quota.loc[order.index[:remaining]] += 1
    if int(quota.sum()) != total:
        raise RuntimeError("Scenario largest-remainder allocation lost people.")
    return quota.astype("int64")


def _prepare_plan(baseline_plan, scenario_id, tsunami_arrival_hour):
    """Copy the baseline and retain explicit baseline provenance columns."""
    plan = baseline_plan.copy(deep=True)
    plan["scenario_id"] = scenario_id
    plan["scenario_plan_id"] = scenario_id + ":" + plan["plan_id"].astype(str)
    plan["baseline_evacuation_mode"] = plan["evacuation_mode"].astype(str)
    plan["baseline_chosen_destination_id"] = plan[
        "chosen_destination_id"
    ].astype(str)
    plan["baseline_chosen_distance_m"] = pd.to_numeric(
        plan["chosen_distance_m"], errors="coerce"
    )
    plan["baseline_response_delay_min"] = pd.to_numeric(
        plan["response_delay_min"], errors="coerce"
    )
    plan["scenario_added_response_delay_min"] = 0.0
    plan["scenario_tsunami_arrival_hour"] = float(tsunami_arrival_hour)
    plan["scenario_gate_policy"] = "baseline_deep_safe_gate_allocation"
    plan["scenario_gate_assignment_limit"] = pd.NA
    plan["scenario_gate_assignment_unit"] = pd.NA
    plan["scenario_closed_gate_id"] = pd.NA
    plan["scenario_assignment_id"] = plan["assignment_id"].astype(str)
    plan["scenario_change_reason"] = "Baseline reference plan"
    plan["scenario_mode_transition"] = "unchanged"
    plan["vehicle_proxy_class"] = np.where(
        plan["evacuation_mode"].eq("vehicle_proxy"),
        "DAS_inferred_vehicle_proxy",
        pd.NA,
    )
    return plan


def _refresh_timing(plan, alert_time_hour, tsunami_arrival_hour, time_bin_min, walk_speed_m_per_min):
    """Refresh departure bins and walking-only free-flow timing diagnostics."""
    plan["response_delay_min"] = pd.to_numeric(
        plan["response_delay_min"], errors="coerce"
    ).fillna(0.0)
    plan["planned_departure_time_s"] = (
        plan["response_delay_min"] * 60.0
    ).round().astype("int64")
    plan["planned_departure_time_hour"] = (
        float(alert_time_hour) + plan["response_delay_min"] / 60.0
    )
    walk_mask = plan["evacuation_mode"].eq("walk")
    plan["estimated_walk_travel_time_min"] = np.where(
        walk_mask,
        pd.to_numeric(plan["chosen_distance_m"], errors="coerce")
        / float(walk_speed_m_per_min),
        np.nan,
    )
    plan["estimated_arrival_time_hour"] = (
        plan["planned_departure_time_hour"]
        + plan["estimated_walk_travel_time_min"] / 60.0
    )
    plan["estimated_arrives_before_tsunami"] = pd.Series(
        pd.NA, index=plan.index, dtype="boolean"
    )
    plan.loc[walk_mask, "estimated_arrives_before_tsunami"] = (
        plan.loc[walk_mask, "estimated_arrival_time_hour"]
        <= float(tsunami_arrival_hour)
    ).to_numpy()
    bin_seconds = int(time_bin_min) * 60
    plan["departure_time_bin_s"] = (
        plan["planned_departure_time_s"] // bin_seconds * bin_seconds
    ).astype("int64")
    return plan


def _promote_walkers_to_vehicle_proxy(plan, scenario_id, target_vehicle_share):
    """Promote a deterministic, spatially proportional subset of walkers."""
    total_people = len(plan)
    existing_vehicle_people = int(plan["evacuation_mode"].eq("vehicle_proxy").sum())
    target_vehicle_people = int(round(float(target_vehicle_share) * total_people))
    promotion_count = target_vehicle_people - existing_vehicle_people
    if promotion_count < 0:
        raise RuntimeError("The high-vehicle target is below the baseline share.")

    eligible_mask = (
        plan["evacuation_mode"].eq("walk")
        & ~plan["alert_state"].eq("in_transit")
    )
    eligible = plan.loc[eligible_mask].copy()
    if promotion_count > len(eligible):
        raise RuntimeError(
            "There are not enough non-transit walkers for the requested "
            "high-vehicle sensitivity scenario."
        )

    group_columns = ["origin_id", "chosen_destination_id"]
    eligible["_scenario_group"] = (
        eligible[group_columns].astype(str).agg("|".join, axis=1)
    )
    group_counts = eligible.groupby("_scenario_group", sort=True).size()
    group_quota = _largest_remainder_quotas(
        group_counts,
        promotion_count,
        tie_breakers=group_counts.index.astype(str),
    )

    promoted_indices = []
    for group_id, quota in group_quota.items():
        if int(quota) == 0:
            continue
        group = eligible.loc[eligible["_scenario_group"].eq(group_id)].copy()
        group["_scenario_hash"] = group["person_id"].map(
            lambda value: _stable_hash(value, f"{scenario_id}|vehicle_promotion")
        )
        group = group.sort_values(["_scenario_hash", "person_id"], kind="stable")
        promoted_indices.extend(group.head(int(quota)).index.tolist())

    if len(promoted_indices) != promotion_count:
        raise RuntimeError("High-vehicle promotion did not reach its target.")

    plan.loc[promoted_indices, "evacuation_mode"] = "vehicle_proxy"
    plan.loc[promoted_indices, "vehicle_proxy_id"] = (
        "SVP_" + plan.loc[promoted_indices, "person_id"].astype(str)
    )
    plan.loc[promoted_indices, "vehicle_proxy_class"] = (
        "scenario_promoted_private_vehicle"
    )
    plan.loc[promoted_indices, "mode_decision_reason"] = (
        "High-vehicle sensitivity: deterministic walk-to-vehicle proxy conversion"
    )
    plan.loc[promoted_indices, "scenario_mode_transition"] = "walk_to_vehicle_proxy"
    plan.loc[promoted_indices, "scenario_change_reason"] = (
        "Vehicle-use sensitivity: promoted from walking while preserving origin and destination"
    )
    plan.loc[promoted_indices, "plan_status"] = (
        "planned_vehicle_requires_Aimsun_simulation"
    )

    observed_vehicle_people = int(plan["evacuation_mode"].eq("vehicle_proxy").sum())
    if observed_vehicle_people != target_vehicle_people:
        raise RuntimeError("High-vehicle scenario did not preserve its exact target.")
    return plan, promotion_count


def _gate_route_cache(origin_id, origin_snap_by_id, routes_to_safe_exit_gates, cache):
    if origin_id not in cache:
        if origin_id not in origin_snap_by_id.index:
            raise RuntimeError(f"Origin {origin_id} has no road-network snap.")
        cache[origin_id] = {
            route["gate_id"]: route
            for route in routes_to_safe_exit_gates(origin_snap_by_id.loc[origin_id])
        }
    return cache[origin_id]


def _reassign_gate_rows(
    plan,
    row_mask,
    scenario_id,
    policy_name,
    active_gate_ids,
    assignment_limits,
    assignment_unit,
    physical_gate_capacity,
    gate_table,
    origin_snap_by_id,
    routes_to_safe_exit_gates,
):
    """Reassign selected people with the existing regret-greedy network logic.

    The allocator is deterministic and capacity constrained.  It minimizes
    avoidable detours greedily; it is deliberately labelled as a heuristic,
    rather than claiming mathematical global optimality.
    """
    selected_indices = plan.index[row_mask].tolist()
    if not selected_indices:
        return plan, pd.DataFrame()

    active_gate_ids = [str(gate_id) for gate_id in active_gate_ids]
    assignment_limits = {
        str(gate_id): int(limit)
        for gate_id, limit in assignment_limits.items()
    }
    if set(active_gate_ids) != set(assignment_limits):
        raise RuntimeError("Scenario gate limits do not match the active gates.")
    if sum(assignment_limits.values()) < len(selected_indices):
        raise RuntimeError("Scenario gate limits cannot accommodate selected demand.")

    route_cache = {}
    selected = plan.loc[selected_indices, ["origin_id", "person_id"]].copy()
    selected["_scenario_hash"] = selected["person_id"].map(
        lambda value: _stable_hash(value, f"{scenario_id}|{policy_name}|gate_order")
    )
    pending = []
    for source_order, (origin_id, group) in enumerate(
        selected.groupby("origin_id", sort=True)
    ):
        route_by_gate = _gate_route_cache(
            origin_id,
            origin_snap_by_id,
            routes_to_safe_exit_gates,
            route_cache,
        )
        routes = [
            route_by_gate[gate_id]
            for gate_id in active_gate_ids
            if gate_id in route_by_gate
        ]
        routes = sorted(routes, key=lambda route: (route["route_distance_m"], route["gate_id"]))
        if not routes:
            raise RuntimeError(
                f"No active safe gate is reachable from origin {origin_id}."
            )
        ordered_indices = group.sort_values(
            ["_scenario_hash", "person_id"], kind="stable"
        ).index.tolist()
        pending.append(
            {
                "origin_id": origin_id,
                "source_order": source_order,
                "row_indices": ordered_indices,
                "routes": routes,
                "part_number": 0,
            }
        )

    remaining = assignment_limits.copy()
    allocation_records = []
    gate_lookup = gate_table.set_index("gate_id", drop=False)
    sequence = 0
    while True:
        active_pending = [item for item in pending if item["row_indices"]]
        if not active_pending:
            break

        candidates = []
        for item in active_pending:
            viable = [
                route
                for route in item["routes"]
                if remaining[route["gate_id"]] > 0
            ]
            if not viable:
                raise RuntimeError(
                    "No remaining scenario gate capacity is reachable for "
                    f"origin {item['origin_id']}."
                )
            best_route = viable[0]
            regret_m = (
                viable[1]["route_distance_m"] - best_route["route_distance_m"]
                if len(viable) > 1
                else float("inf")
            )
            candidates.append((item, best_route, regret_m))

        item, route, regret_m = sorted(
            candidates,
            key=lambda candidate: (
                -candidate[2],
                -len(candidate[0]["row_indices"]),
                str(candidate[0]["origin_id"]),
                candidate[0]["source_order"],
            ),
        )[0]
        gate_id = route["gate_id"]
        assigned_count = min(len(item["row_indices"]), remaining[gate_id])
        if assigned_count <= 0:
            raise RuntimeError("Scenario gate allocation made no progress.")
        assigned_indices = item["row_indices"][:assigned_count]
        item["row_indices"] = item["row_indices"][assigned_count:]
        remaining[gate_id] -= assigned_count
        gate = gate_lookup.loc[gate_id]

        plan.loc[assigned_indices, "chosen_destination_type"] = "outside_flood_area"
        plan.loc[assigned_indices, "chosen_destination_id"] = gate_id
        plan.loc[assigned_indices, "chosen_distance_m"] = float(route["route_distance_m"])
        plan.loc[assigned_indices, "network_path_distance_m"] = float(route["network_distance_m"])
        plan.loc[assigned_indices, "origin_network_access_m"] = float(route["origin_access_m"])
        plan.loc[assigned_indices, "destination_network_access_m"] = 0.0
        plan.loc[assigned_indices, "origin_network_node_id"] = int(route["origin_network_node_id"])
        plan.loc[assigned_indices, "destination_network_node_id"] = int(
            route["destination_network_node_id"]
        )
        plan.loc[assigned_indices, "destination_capacity_people"] = float(
            physical_gate_capacity[gate_id]
        )
        plan.loc[assigned_indices, "destination_x_itm"] = float(gate.geometry.x)
        plan.loc[assigned_indices, "destination_y_itm"] = float(gate.geometry.y)
        plan.loc[assigned_indices, "scenario_gate_policy"] = policy_name
        plan.loc[assigned_indices, "scenario_gate_assignment_limit"] = int(
            assignment_limits[gate_id]
        )
        plan.loc[assigned_indices, "scenario_gate_assignment_unit"] = assignment_unit
        plan.loc[assigned_indices, "scenario_assignment_id"] = (
            plan.loc[assigned_indices, "assignment_id"].astype(str)
            + "__"
            + scenario_id
            + "__"
            + gate_id
        )
        plan.loc[assigned_indices, "scenario_change_reason"] = (
            f"{policy_name}: capacity-constrained safe-gate reassignment"
        )

        allocation_records.append(
            {
                "scenario_id": scenario_id,
                "policy_name": policy_name,
                "assignment_unit": assignment_unit,
                "assignment_sequence": sequence,
                "origin_id": item["origin_id"],
                "gate_id": gate_id,
                "assigned_count": int(assigned_count),
                "route_distance_m": float(route["route_distance_m"]),
                "regret_m": float(regret_m),
                "remaining_assignment_limit": int(remaining[gate_id]),
            }
        )
        item["part_number"] += 1
        sequence += 1

    plan["scenario_reassigned_from_destination_id"] = plan[
        "baseline_chosen_destination_id"
    ]
    plan["scenario_destination_changed"] = (
        plan["chosen_destination_id"].astype(str)
        != plan["baseline_chosen_destination_id"].astype(str)
    )
    return plan, pd.DataFrame(allocation_records)


def _make_pedestrian_od(plan):
    group_columns = [
        "scenario_id",
        "departure_time_bin_s",
        "origin_id",
        "chosen_destination_id",
        "chosen_destination_type",
        "origin_x_itm",
        "origin_y_itm",
        "destination_x_itm",
        "destination_y_itm",
    ]
    return (
        plan.loc[plan["evacuation_mode"].eq("walk")]
        .groupby(group_columns, dropna=False, as_index=False)
        .agg(
            pedestrian_count=("scenario_plan_id", "size"),
            mean_route_distance_m=("chosen_distance_m", "mean"),
            mean_response_delay_min=("response_delay_min", "mean"),
        )
        .sort_values(
            ["scenario_id", "departure_time_bin_s", "origin_id", "chosen_destination_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _make_vehicle_od(plan, vehicle_occupancy_proxy):
    vehicle_plan = plan.loc[plan["evacuation_mode"].eq("vehicle_proxy")].copy()
    vehicle_plan["vehicle_proxy_count"] = 1.0 / float(vehicle_occupancy_proxy)
    vehicle_plan["aimsun_vehicle_type"] = np.where(
        vehicle_plan["vehicle_proxy_class"].eq("scenario_promoted_private_vehicle"),
        "EMERGENCY_PRIVATE_VEHICLE_SENSITIVITY",
        "TO_BE_MAPPED_from_DAS_stop_mode",
    )
    vehicle_plan["source_stop_mode"] = (
        vehicle_plan["source_stop_mode"].fillna("Unknown").astype(str)
    )
    group_columns = [
        "scenario_id",
        "departure_time_bin_s",
        "origin_id",
        "chosen_destination_id",
        "chosen_destination_type",
        "origin_x_itm",
        "origin_y_itm",
        "destination_x_itm",
        "destination_y_itm",
        "source_stop_mode",
        "vehicle_proxy_class",
        "aimsun_vehicle_type",
    ]
    vehicle_od = (
        vehicle_plan.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            vehicle_count=("vehicle_proxy_count", "sum"),
            represented_people=("scenario_plan_id", "size"),
            mean_route_distance_m=("chosen_distance_m", "mean"),
            mean_response_delay_min=("response_delay_min", "mean"),
        )
        .sort_values(
            ["scenario_id", "departure_time_bin_s", "origin_id", "chosen_destination_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return vehicle_od


def _make_validation(
    plan,
    baseline_person_ids,
    pedestrian_od,
    vehicle_od,
    gate_table,
    physical_gate_capacity,
    scenario_id,
    closed_gate_id=None,
):
    gate_ids = gate_table["gate_id"].astype(str).tolist()
    gate_mask = plan["chosen_destination_id"].astype(str).isin(gate_ids)
    gate_loads = (
        plan.loc[gate_mask]
        .groupby("chosen_destination_id")
        .size()
        .reindex(gate_ids, fill_value=0)
        .astype("int64")
    )
    gate_cap_series = pd.Series(physical_gate_capacity).reindex(gate_ids).astype(int)
    vehicle_people = int(plan["evacuation_mode"].eq("vehicle_proxy").sum())
    pedestrian_people = int(plan["evacuation_mode"].eq("walk").sum())
    bin_seconds = 300
    checks = [
        (
            "person_count_preserved",
            len(plan) == len(baseline_person_ids),
            len(plan),
            len(baseline_person_ids),
        ),
        (
            "same_unique_person_ids_as_baseline",
            set(plan["person_id"]) == baseline_person_ids and not plan["person_id"].duplicated().any(),
            int(plan["person_id"].nunique()),
            len(baseline_person_ids),
        ),
        (
            "one_scenario_id",
            plan["scenario_id"].nunique() == 1 and plan["scenario_id"].iloc[0] == scenario_id,
            plan["scenario_id"].nunique(),
            1,
        ),
        (
            "origin_destination_coordinates_complete",
            not plan[["origin_x_itm", "origin_y_itm", "destination_x_itm", "destination_y_itm"]].isna().any().any(),
            int(plan[["origin_x_itm", "origin_y_itm", "destination_x_itm", "destination_y_itm"]].isna().sum().sum()),
            0,
        ),
        (
            "departure_bins_match_plan",
            (plan["departure_time_bin_s"] == (plan["planned_departure_time_s"] // bin_seconds * bin_seconds)).all(),
            int((plan["departure_time_bin_s"] != (plan["planned_departure_time_s"] // bin_seconds * bin_seconds)).sum()),
            0,
        ),
        (
            "modal_od_totals_match_plan",
            int(pedestrian_od["pedestrian_count"].sum()) == pedestrian_people
            and int(round(float(vehicle_od["represented_people"].sum()))) == vehicle_people,
            int(pedestrian_od["pedestrian_count"].sum() + vehicle_od["represented_people"].sum()),
            len(plan),
        ),
        (
            "physical_safe_gate_capacity_respected",
            (gate_loads <= gate_cap_series).all(),
            int((gate_loads - gate_cap_series).max()),
            0,
        ),
    ]
    if closed_gate_id is not None:
        checks.append(
            (
                "closed_gate_has_no_assigned_people",
                int(gate_loads.loc[str(closed_gate_id)]) == 0,
                int(gate_loads.loc[str(closed_gate_id)]),
                0,
            )
        )
    validation = pd.DataFrame(
        checks, columns=["check", "passed", "observed", "expected"]
    )
    validation.insert(0, "scenario_id", scenario_id)
    return validation, gate_loads


def _add_centroid_labels(od, mapping_candidates):
    if mapping_candidates is None:
        return None
    origins = mapping_candidates.loc[
        mapping_candidates["entity_role"].eq("origin"),
        ["entity_id", "proposed_centroid_label"],
    ].rename(
        columns={
            "entity_id": "origin_id",
            "proposed_centroid_label": "origin_centroid_label",
        }
    )
    destinations = mapping_candidates.loc[
        ~mapping_candidates["entity_role"].eq("origin"),
        ["entity_id", "proposed_centroid_label"],
    ].rename(
        columns={
            "entity_id": "chosen_destination_id",
            "proposed_centroid_label": "destination_centroid_label",
        }
    )
    labelled = (
        od.merge(origins, on="origin_id", how="left", validate="many_to_one")
        .merge(
            destinations,
            on="chosen_destination_id",
            how="left",
            validate="many_to_one",
        )
    )
    if labelled[["origin_centroid_label", "destination_centroid_label"]].isna().any().any():
        raise RuntimeError("A scenario OD row has no candidate Aimsun centroid label.")
    labelled["aimsun_import_status"] = "candidate_labels_only_until_model_mapping_is_completed"
    return labelled


def _scenario_summary(plan, gate_loads, scenario_id):
    walk_mask = plan["evacuation_mode"].eq("walk")
    walk_arrives = plan.loc[walk_mask, "estimated_arrives_before_tsunami"].fillna(False)
    return {
        "scenario_id": scenario_id,
        "total_people": int(len(plan)),
        "pedestrian_people": int(walk_mask.sum()),
        "vehicle_proxy_people": int(plan["evacuation_mode"].eq("vehicle_proxy").sum()),
        "vehicle_proxy_share": round(float(plan["evacuation_mode"].eq("vehicle_proxy").mean()), 6),
        "vertical_refuge_people": int(plan["chosen_destination_type"].eq("vertical_refuge_building").sum()),
        "safe_gate_people": int(plan["chosen_destination_id"].isin(gate_loads.index).sum()),
        "safe_gates_in_use": int((gate_loads > 0).sum()),
        "max_safe_gate_load_people": int(gate_loads.max()),
        "walking_people_estimated_before_tsunami": int(walk_arrives.sum()),
        "walking_people_estimated_after_tsunami": int((~walk_arrives).sum()),
        "scenario_tsunami_arrival_hour": float(
            plan["scenario_tsunami_arrival_hour"].iloc[0]
        ),
        "mean_response_delay_min": round(float(plan["response_delay_min"].mean()), 3),
        "mean_route_distance_m": round(float(pd.to_numeric(plan["chosen_distance_m"], errors="coerce").mean()), 3),
    }


def build_evacuation_scenarios(
    *,
    baseline_evacuation_plan,
    safe_exit_gates,
    origin_snap_by_id,
    routes_to_safe_exit_gates,
    output_dir,
    alert_time_hour,
    tsunami_arrival_hour,
    time_bin_min,
    walk_speed_m_per_min,
    vehicle_occupancy_proxy,
    write_csv_with_fallback,
    write_parquet_with_fallback,
    aimsun_mapping_candidates=None,
):
    """Create Phase 4 scenario plans, ODs, controls, summaries, and QA files."""
    required_columns = {
        "plan_id", "person_id", "origin_id", "chosen_destination_id",
        "chosen_destination_type", "evacuation_mode", "response_delay_min",
        "origin_x_itm", "origin_y_itm", "destination_x_itm", "destination_y_itm",
    }
    missing_columns = sorted(required_columns - set(baseline_evacuation_plan.columns))
    if missing_columns:
        raise RuntimeError(
            "Baseline plan is missing columns required for Phase 4: "
            + ", ".join(missing_columns)
        )
    if baseline_evacuation_plan["person_id"].duplicated().any():
        raise RuntimeError("Baseline plan must contain one row per person.")

    scenario_root = Path(output_dir) / "scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)
    baseline_person_ids = set(baseline_evacuation_plan["person_id"])
    baseline_fingerprint = pd.util.hash_pandas_object(
        baseline_evacuation_plan, index=True
    ).sum()

    gate_table = safe_exit_gates.copy()
    gate_table["gate_id"] = gate_table["gate_id"].astype(str)
    physical_gate_capacity = gate_table.set_index("gate_id")[
        "planning_capacity_people"
    ].astype(int).to_dict()
    gate_ids = gate_table["gate_id"].tolist()
    if len(gate_ids) != 8 or len(set(gate_ids)) != len(gate_ids):
        raise RuntimeError("The scenario generator expects eight unique safe gates.")

    baseline_vehicle_people = int(
        baseline_evacuation_plan["evacuation_mode"].eq("vehicle_proxy").sum()
    )
    baseline_total_people = len(baseline_evacuation_plan)
    high_vehicle_target_share = 0.60
    managed_vehicle_cap_per_gate = int(
        baseline_evacuation_plan.loc[
            baseline_evacuation_plan["evacuation_mode"].eq("vehicle_proxy")
            & baseline_evacuation_plan["chosen_destination_id"].astype(str).isin(gate_ids)
        ].shape[0]
        / len(gate_ids)
    )
    closed_gate_id = "SAFE_GATE_07"
    if closed_gate_id not in gate_ids:
        raise RuntimeError(f"Configured closure gate {closed_gate_id} is missing.")

    scenario_config = pd.DataFrame([
        {
            "scenario_id": "baseline_reference_v1",
            "scenario_type": "reference",
            "description": "Existing person-level baseline without a new change.",
            "response_delay_change": "none",
            "target_vehicle_share": baseline_vehicle_people / baseline_total_people,
            "safe_gate_logic": "existing deep-safe-gate allocation",
            "closed_safe_gate_id": pd.NA,
            "tsunami_arrival_hour": float(tsunami_arrival_hour),
            "Aimsun_network_control_status": "not_required_for_OD_generation",
        },
        {
            "scenario_id": "slow_response_plus_10min_v1",
            "scenario_type": "response_sensitivity",
            "description": "Add ten minutes to every baseline response delay.",
            "response_delay_change": "+10 minutes for every person",
            "target_vehicle_share": baseline_vehicle_people / baseline_total_people,
            "safe_gate_logic": "existing deep-safe-gate allocation",
            "closed_safe_gate_id": pd.NA,
            "tsunami_arrival_hour": float(tsunami_arrival_hour),
            "Aimsun_network_control_status": "not_required_for_OD_generation",
        },
        {
            "scenario_id": "high_vehicle_use_60pct_v1",
            "scenario_type": "mode_choice_sensitivity",
            "description": "Increase vehicle-proxy share from the DAS-derived baseline to 60%.",
            "response_delay_change": "none",
            "target_vehicle_share": high_vehicle_target_share,
            "safe_gate_logic": "existing deep-safe-gate allocation",
            "closed_safe_gate_id": pd.NA,
            "tsunami_arrival_hour": float(tsunami_arrival_hour),
            "Aimsun_network_control_status": "not_required_for_OD_generation",
        },
        {
            "scenario_id": "managed_vehicle_gate_balance_v1",
            "scenario_type": "managed_evacuation",
            "description": "Reassign only vehicle proxies sent to gates, with an equal vehicle cap at each gate.",
            "response_delay_change": "none",
            "target_vehicle_share": baseline_vehicle_people / baseline_total_people,
            "safe_gate_logic": f"capacity-constrained heuristic; {managed_vehicle_cap_per_gate} vehicle proxies per gate",
            "closed_safe_gate_id": pd.NA,
            "tsunami_arrival_hour": float(tsunami_arrival_hour),
            "Aimsun_network_control_status": "pending_Aimsun_model_controls",
        },
        {
            "scenario_id": "gate_07_closure_v1",
            "scenario_type": "infrastructure_failure",
            "description": "Close SAFE_GATE_07 and reallocate every outside-flood evacuee among seven gates.",
            "response_delay_change": "none",
            "target_vehicle_share": baseline_vehicle_people / baseline_total_people,
            "safe_gate_logic": "capacity-constrained heuristic; seven active gates at 4,000 people each",
            "closed_safe_gate_id": closed_gate_id,
            "tsunami_arrival_hour": float(tsunami_arrival_hour),
            "Aimsun_network_control_status": "pending_Aimsun_model_controls",
        },
        {
            "scenario_id": "short_warning_60min_v1",
            "scenario_type": "warning_window_sensitivity",
            "description": "Shorten the warning window to 60 minutes; OD demand remains unchanged.",
            "response_delay_change": "none",
            "target_vehicle_share": baseline_vehicle_people / baseline_total_people,
            "safe_gate_logic": "existing deep-safe-gate allocation",
            "closed_safe_gate_id": pd.NA,
            "tsunami_arrival_hour": float(alert_time_hour) + 1.0,
            "Aimsun_network_control_status": "not_required_for_OD_generation",
        },
        {
            "scenario_id": "walking_only_extreme_v1",
            "scenario_type": "mode_choice_extreme",
            "description": "Extreme sensitivity: every evacuee walks and no evacuation vehicle demand is created.",
            "response_delay_change": "none",
            "target_vehicle_share": 0.0,
            "safe_gate_logic": "existing deep-safe-gate allocation",
            "closed_safe_gate_id": pd.NA,
            "tsunami_arrival_hour": float(tsunami_arrival_hour),
            "Aimsun_network_control_status": "requires_pedestrian_model_for_operational_simulation",
        },
    ])

    control_plan = pd.DataFrame([
        {
            "scenario_id": "managed_vehicle_gate_balance_v1",
            "control_id": "CTRL_GATE_GUIDANCE_01",
            "control_status": "pending_Aimsun_model_controls",
            "target_population": "vehicle proxies assigned to safe gates",
            "policy_action": "Apply the scenario-specific centroid-to-gate guidance and prioritize eastbound evacuation flow.",
            "start_time_s": 0,
            "end_time_s": pd.NA,
            "implementation_note": "Requires verified Aimsun section, turn, signal, and connector IDs; not applied by this notebook.",
        },
        {
            "scenario_id": "managed_vehicle_gate_balance_v1",
            "control_id": "CTRL_KEEP_OUT_OF_HAZARD_01",
            "control_status": "pending_background_demand_and_Aimsun_controls",
            "target_population": "nearby non-evacuation traffic outside the flood area",
            "policy_action": "Discourage or prohibit westbound entry toward the flood area during evacuation.",
            "start_time_s": 0,
            "end_time_s": pd.NA,
            "implementation_note": "The current demand plan contains only people inside the hazard area, so this is staged rather than simulated.",
        },
        {
            "scenario_id": "gate_07_closure_v1",
            "control_id": "CTRL_CLOSE_SAFE_GATE_07",
            "control_status": "pending_Aimsun_model_controls",
            "target_population": "all evacuation vehicle and pedestrian routes to SAFE_GATE_07",
            "policy_action": "Close the gate connector or its access sections and enforce the rerouted OD assignments.",
            "start_time_s": 0,
            "end_time_s": pd.NA,
            "implementation_note": "Requires the actual Aimsun model topology; the gate is already removed from this scenario OD output.",
        },
    ])

    scenario_plans = {}
    scenario_vehicle_ods = {}
    scenario_pedestrian_ods = {}
    scenario_gate_allocations = {}
    scenario_summaries = []
    scenario_validations = []
    manifest = []

    def finalize(plan, scenario_id, closed_gate=None, allocation_records=None):
        plan = _refresh_timing(
            plan,
            alert_time_hour,
            float(plan["scenario_tsunami_arrival_hour"].iloc[0]),
            time_bin_min,
            walk_speed_m_per_min,
        )
        pedestrian_od = _make_pedestrian_od(plan)
        vehicle_od = _make_vehicle_od(plan, vehicle_occupancy_proxy)
        validation, gate_loads = _make_validation(
            plan,
            baseline_person_ids,
            pedestrian_od,
            vehicle_od,
            gate_table,
            physical_gate_capacity,
            scenario_id,
            closed_gate_id=closed_gate,
        )
        if not validation["passed"].all():
            failed = validation.loc[~validation["passed"], "check"].tolist()
            raise RuntimeError(f"Scenario {scenario_id} failed QA: {failed}")
        scenario_plans[scenario_id] = plan
        scenario_vehicle_ods[scenario_id] = vehicle_od
        scenario_pedestrian_ods[scenario_id] = pedestrian_od
        scenario_gate_allocations[scenario_id] = (
            allocation_records.copy()
            if allocation_records is not None and not allocation_records.empty
            else pd.DataFrame()
        )
        scenario_summaries.append(_scenario_summary(plan, gate_loads, scenario_id))
        scenario_validations.append(validation)

    baseline_plan = _prepare_plan(
        baseline_evacuation_plan, "baseline_reference_v1", tsunami_arrival_hour
    )
    finalize(baseline_plan, "baseline_reference_v1")

    slow_plan = _prepare_plan(
        baseline_evacuation_plan, "slow_response_plus_10min_v1", tsunami_arrival_hour
    )
    slow_plan["scenario_added_response_delay_min"] = 10.0
    slow_plan["response_delay_min"] = slow_plan["baseline_response_delay_min"] + 10.0
    slow_plan["scenario_change_reason"] = "Slow-response sensitivity: +10 minutes before departure"
    finalize(slow_plan, "slow_response_plus_10min_v1")

    high_vehicle_plan = _prepare_plan(
        baseline_evacuation_plan, "high_vehicle_use_60pct_v1", tsunami_arrival_hour
    )
    high_vehicle_plan, promoted_people = _promote_walkers_to_vehicle_proxy(
        high_vehicle_plan, "high_vehicle_use_60pct_v1", high_vehicle_target_share
    )
    finalize(high_vehicle_plan, "high_vehicle_use_60pct_v1")

    managed_plan = _prepare_plan(
        baseline_evacuation_plan, "managed_vehicle_gate_balance_v1", tsunami_arrival_hour
    )
    managed_mask = (
        managed_plan["evacuation_mode"].eq("vehicle_proxy")
        & managed_plan["chosen_destination_id"].astype(str).isin(gate_ids)
    )
    managed_limits = {gate_id: managed_vehicle_cap_per_gate for gate_id in gate_ids}
    managed_plan, managed_allocations = _reassign_gate_rows(
        managed_plan,
        managed_mask,
        "managed_vehicle_gate_balance_v1",
        "equal_vehicle_gate_balance_heuristic",
        gate_ids,
        managed_limits,
        "vehicle_proxy_people",
        physical_gate_capacity,
        gate_table,
        origin_snap_by_id,
        routes_to_safe_exit_gates,
    )
    finalize(
        managed_plan,
        "managed_vehicle_gate_balance_v1",
        allocation_records=managed_allocations,
    )

    closure_plan = _prepare_plan(
        baseline_evacuation_plan, "gate_07_closure_v1", tsunami_arrival_hour
    )
    closure_plan["scenario_closed_gate_id"] = closed_gate_id
    closure_mask = closure_plan["chosen_destination_id"].astype(str).isin(gate_ids)
    active_closure_gates = [gate_id for gate_id in gate_ids if gate_id != closed_gate_id]
    closure_limits = {gate_id: physical_gate_capacity[gate_id] for gate_id in active_closure_gates}
    closure_plan, closure_allocations = _reassign_gate_rows(
        closure_plan,
        closure_mask,
        "gate_07_closure_v1",
        "gate_07_closure_reallocation_heuristic",
        active_closure_gates,
        closure_limits,
        "people",
        physical_gate_capacity,
        gate_table,
        origin_snap_by_id,
        routes_to_safe_exit_gates,
    )
    finalize(
        closure_plan,
        "gate_07_closure_v1",
        closed_gate=closed_gate_id,
        allocation_records=closure_allocations,
    )

    short_warning_plan = _prepare_plan(
        baseline_evacuation_plan,
        "short_warning_60min_v1",
        float(alert_time_hour) + 1.0,
    )
    short_warning_plan["scenario_change_reason"] = (
        "Short-warning sensitivity: tsunami arrival deadline reduced to one hour after alert"
    )
    finalize(short_warning_plan, "short_warning_60min_v1")

    walking_only_plan = _prepare_plan(
        baseline_evacuation_plan, "walking_only_extreme_v1", tsunami_arrival_hour
    )
    vehicle_mask = walking_only_plan["evacuation_mode"].eq("vehicle_proxy")
    walking_only_plan.loc[vehicle_mask, "evacuation_mode"] = "walk"
    walking_only_plan.loc[vehicle_mask, "vehicle_proxy_id"] = pd.NA
    walking_only_plan.loc[vehicle_mask, "vehicle_proxy_class"] = pd.NA
    walking_only_plan.loc[vehicle_mask, "mode_decision_reason"] = (
        "Walking-only extreme sensitivity: vehicle proxy removed"
    )
    walking_only_plan.loc[vehicle_mask, "scenario_mode_transition"] = (
        "vehicle_proxy_to_walk"
    )
    walking_only_plan.loc[vehicle_mask, "scenario_change_reason"] = (
        "Walking-only extreme sensitivity: vehicle proxy converted to walking"
    )
    walking_only_plan.loc[vehicle_mask, "plan_status"] = (
        "planned_walk_time_is_a_free_flow_proxy"
    )
    finalize(walking_only_plan, "walking_only_extreme_v1")

    if pd.util.hash_pandas_object(baseline_evacuation_plan, index=True).sum() != baseline_fingerprint:
        raise RuntimeError("Phase 4 modified the baseline plan in memory.")

    scenario_validation = pd.concat(scenario_validations, ignore_index=True)
    scenario_specific_checks = []
    high_vehicle_plan = scenario_plans["high_vehicle_use_60pct_v1"]
    high_vehicle_observed = int(
        high_vehicle_plan["evacuation_mode"].eq("vehicle_proxy").sum()
    )
    high_vehicle_expected = int(round(high_vehicle_target_share * baseline_total_people))
    scenario_specific_checks.append({
        "scenario_id": "high_vehicle_use_60pct_v1",
        "check": "exact_high_vehicle_target",
        "passed": high_vehicle_observed == high_vehicle_expected,
        "observed": high_vehicle_observed,
        "expected": high_vehicle_expected,
    })

    slow_response_plan = scenario_plans["slow_response_plus_10min_v1"]
    slow_response_observed = bool(np.isclose(
        slow_response_plan["response_delay_min"].to_numpy(),
        slow_response_plan["baseline_response_delay_min"].to_numpy() + 10.0,
    ).all())
    scenario_specific_checks.append({
        "scenario_id": "slow_response_plus_10min_v1",
        "check": "response_delay_increased_by_ten_minutes",
        "passed": slow_response_observed,
        "observed": "all people" if slow_response_observed else "mismatch found",
        "expected": "all people",
    })

    managed_vehicle_plan = scenario_plans["managed_vehicle_gate_balance_v1"]
    managed_vehicle_gate_loads = (
        managed_vehicle_plan.loc[
            managed_vehicle_plan["evacuation_mode"].eq("vehicle_proxy")
            & managed_vehicle_plan["chosen_destination_id"].astype(str).isin(gate_ids)
        ]
        .groupby("chosen_destination_id")
        .size()
        .reindex(gate_ids, fill_value=0)
    )
    managed_balance_observed = bool(
        (managed_vehicle_gate_loads == managed_vehicle_cap_per_gate).all()
    )
    scenario_specific_checks.append({
        "scenario_id": "managed_vehicle_gate_balance_v1",
        "check": "equal_vehicle_gate_loads",
        "passed": managed_balance_observed,
        "observed": "; ".join(
            f"{gate_id}={count}"
            for gate_id, count in managed_vehicle_gate_loads.items()
        ),
        "expected": f"{managed_vehicle_cap_per_gate} vehicle proxies at every gate",
    })

    short_warning_plan = scenario_plans["short_warning_60min_v1"]
    short_warning_observed = bool(np.isclose(
        short_warning_plan["scenario_tsunami_arrival_hour"],
        float(alert_time_hour) + 1.0,
    ).all())
    scenario_specific_checks.append({
        "scenario_id": "short_warning_60min_v1",
        "check": "one_hour_warning_window",
        "passed": short_warning_observed,
        "observed": float(short_warning_plan["scenario_tsunami_arrival_hour"].iloc[0]),
        "expected": float(alert_time_hour) + 1.0,
    })

    walking_only_plan = scenario_plans["walking_only_extreme_v1"]
    walking_only_vehicle_people = int(
        walking_only_plan["evacuation_mode"].eq("vehicle_proxy").sum()
    )
    scenario_specific_checks.append({
        "scenario_id": "walking_only_extreme_v1",
        "check": "no_vehicle_proxy_people",
        "passed": walking_only_vehicle_people == 0,
        "observed": walking_only_vehicle_people,
        "expected": 0,
    })

    scenario_validation = pd.concat(
        [scenario_validation, pd.DataFrame(scenario_specific_checks)],
        ignore_index=True,
    )
    if not scenario_validation["passed"].all():
        failed = scenario_validation.loc[~scenario_validation["passed"], [
            "scenario_id", "check"
        ]].to_dict("records")
        raise RuntimeError(f"Scenario-specific QA failed: {failed}")

    scenario_summary = pd.DataFrame(scenario_summaries).merge(
        scenario_config,
        on="scenario_id",
        how="left",
        validate="one_to_one",
    ).sort_values("scenario_id", kind="stable")
    all_vehicle_od = pd.concat(scenario_vehicle_ods.values(), ignore_index=True)
    all_pedestrian_od = pd.concat(scenario_pedestrian_ods.values(), ignore_index=True)
    all_gate_allocations = pd.concat(
        [frame for frame in scenario_gate_allocations.values() if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in scenario_gate_allocations.values()) else pd.DataFrame()

    labelled_vehicle_od = _add_centroid_labels(all_vehicle_od, aimsun_mapping_candidates)
    labelled_pedestrian_od = _add_centroid_labels(all_pedestrian_od, aimsun_mapping_candidates)

    def write_csv(frame, path, output_name, scenario_id=None):
        written_path, status = write_csv_with_fallback(frame, path)
        manifest.append(
            {
                "scenario_id": scenario_id if scenario_id is not None else "all_scenarios",
                "output": output_name,
                "path": str(written_path),
                "status": status,
                "rows": len(frame),
            }
        )
        return written_path

    write_csv(scenario_config, scenario_root / "scenario_config.csv", "scenario_config")
    write_csv(scenario_summary, scenario_root / "scenario_summary.csv", "scenario_summary")
    write_csv(scenario_validation, scenario_root / "scenario_validation.csv", "scenario_validation")
    write_csv(control_plan, scenario_root / "network_control_plan.csv", "network_control_plan")
    write_csv(all_vehicle_od, scenario_root / "vehicle_od_all_scenarios.csv", "vehicle_od_all_scenarios")
    write_csv(all_pedestrian_od, scenario_root / "pedestrian_od_all_scenarios.csv", "pedestrian_od_all_scenarios")
    if not all_gate_allocations.empty:
        write_csv(all_gate_allocations, scenario_root / "gate_reassignment_audit.csv", "gate_reassignment_audit")
    if labelled_vehicle_od is not None:
        write_csv(
            labelled_vehicle_od,
            scenario_root / "aimsun_vehicle_od_centroid_candidates_all_scenarios.csv",
            "aimsun_vehicle_od_centroid_candidates_all_scenarios",
        )
        write_csv(
            labelled_pedestrian_od,
            scenario_root / "aimsun_pedestrian_od_centroid_candidates_all_scenarios.csv",
            "aimsun_pedestrian_od_centroid_candidates_all_scenarios",
        )

    for scenario_id, plan in scenario_plans.items():
        scenario_dir = scenario_root / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        plan_path = write_parquet_with_fallback(plan, scenario_dir / "evacuation_plan.parquet")
        manifest.append(
            {
                "scenario_id": scenario_id,
                "output": "evacuation_plan_parquet",
                "path": str(plan_path),
                "status": "written_or_fallback",
                "rows": len(plan),
            }
        )
        write_csv(
            scenario_vehicle_ods[scenario_id],
            scenario_dir / "vehicle_od.csv",
            "vehicle_od",
            scenario_id,
        )
        write_csv(
            scenario_pedestrian_ods[scenario_id],
            scenario_dir / "pedestrian_od.csv",
            "pedestrian_od",
            scenario_id,
        )
        scenario_validation_frame = scenario_validation.loc[
            scenario_validation["scenario_id"].eq(scenario_id)
        ]
        write_csv(
            scenario_validation_frame,
            scenario_dir / "validation.csv",
            "validation",
            scenario_id,
        )

    scenario_manifest = pd.DataFrame(manifest)
    write_csv(scenario_manifest, scenario_root / "scenario_export_manifest.csv", "scenario_export_manifest")

    return {
        "scenario_config": scenario_config,
        "scenario_summary": scenario_summary,
        "scenario_validation": scenario_validation,
        "scenario_control_plan": control_plan,
        "scenario_plans": scenario_plans,
        "scenario_vehicle_od": all_vehicle_od,
        "scenario_pedestrian_od": all_pedestrian_od,
        "scenario_gate_reassignment_audit": all_gate_allocations,
        "scenario_export_manifest": scenario_manifest,
        "high_vehicle_promoted_people": promoted_people,
        "managed_vehicle_cap_per_gate": managed_vehicle_cap_per_gate,
        "scenario_root": scenario_root,
    }
