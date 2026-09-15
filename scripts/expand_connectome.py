"""
Connectome Expansion Script - Query NeuPrint for additional neurons connected to the
existing 261-neuron CX heading circuit.
"""

import json
import time
import sys
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding='utf-8')

TOKEN = "ddbee0f25b59d207e1e5ede8108a823e8dfa5b9e3d3eb194239dba80cf44f3a4"
BASE = "https://neuprint.janelia.org"
DATASET = "hemibrain:v1.2.1"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Path to existing circuit
CX_FILE = Path(__file__).parent.parent / "data" / "processed" / "cx_heading_v1.json"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "phase7b"
OUTPUT_FILE = OUTPUT_DIR / "connectome_expansion_audit.json"

# Biologically relevant cell types for CX expansion
# These are types known to interact with the central complex circuit
RELEVANT_TYPES = [
    # Motor control / descending
    "DN1", "DN2", "DN3", "DN4", "DN5", "DN6", "DN7", "DN8", "DN9", "DN10",
    "DN11", "DN12", "DN13", "DN14", "DN15", "DN16", "DN17", "DN18", "DN19", "DN20",
    "DNa01", "DNa02", "DNa03", "DNa04", "DNa05", "DNa06", "DNa07", "DNa08", "DNa09", "DNa10",
    "DNb01", "DNb02", "DNb03", "DNb04", "DNb05",
    "DNg01", "DNg02", "DNg03", "DNg04", "DNg05",
    "PFL1", "PFL2", "PFL3", "PFL4", "PFL5",
    "PFLa", "PFLb",
    # Fan-shaped body (FB) - beyond PFNd/PFNv
    "FB1A", "FB1B", "FB1C", "FB1D", "FB1E",
    "FB2A", "FB2B", "FB2C", "FB2D", "FB2E",
    "FB3A", "FB3B", "FB3C", "FB3D", "FB3E",
    "FB4A", "FB4B", "FB4C", "FB4D", "FB4E",
    "FB5A", "FB5B", "FB5C", "FB5D", "FB5E",
    "FB6A", "FB6B", "FB6C", "FB6D", "FB6E",
    "FC1", "FC2", "FC3", "FC4", "FC5",
    "FLA", "FLB",
    # Ellipsoid body (EB) - beyond EPG/EL
    "EPG", "EL", "ET", "EPS", "EPLD",
    "EBa", "EBb",
    # Protocerebral bridge (PB) - beyond Delta7
    "Delta7", "PFL1", "PFL2", "PFL3", "PFL4", "PFL5",
    "PBL", "PBR",
    # Other CX-related circuits
    "EPHP", "EPHI", "EPG",
    "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)",
    "ER2a", "ER2m", "ER2r", "ER4a", "ER4d", "ER4m", "ER4p", "ER4v",
    "PR1", "PR2", "PR3", "PR4", "PR5",
    "AL1", "AL2",
    "LC1", "LC2", "LC3", "LC4", "LC5", "LC6", "LC7", "LC8", "LC9", "LC10",
    "LC11", "LC12", "LC13", "LC14", "LC15", "LC16", "LC17",
    "LHP1", "LHP2",
    "MBON01", "MBON02", "MBON03", "MBON04", "MBON05", "MBON06", "MBON07",
    "MBON08", "MBON09", "MBON10", "MBON11", "MBON12", "MBON13", "MBON14",
    "MBON15", "MBON16", "MBON17", "MBON18", "MBON19", "MBON20",
    "DAN1", "DAN2", "DAN3", "DAN4", "DAN5", "DAN6",
    "DANa1", "DANa2", "DANa3", "DANa4", "DANa5", "DANa6", "DANa7", "DANa8",
    "DANb1", "DANb2", "DANb3",
    "DANg1", "DANg2", "DANg3",
    "PAM1", "PAM2", "PAM3", "PAM4", "PAM5",
    "PPL1", "PPL2",
    # Descending to VNC
    "DNg", "DNa", "DNb",
    # Ventral nerve cord (VNC) motor related
    "aIPg", "aIPs",
    "vMS1", "vMS2",
    "RP1", "RP2", "RP3", "RP4", "RP5",
    "SAG", "SAGa", "SAGb",
    "gTEN",
    "PFGa", "PFGb", "PFGc", "PFGd",
]


def cypher(query):
    """Execute a Cypher query against neuPrint."""
    resp = requests.post(
        f"{BASE}/api/custom/custom",
        headers=HEADERS,
        json={"cypher": query, "dataset": DATASET},
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()


def show(r, limit=10):
    """Show results from a query."""
    cols = r["columns"]
    for row in r["data"][:limit]:
        print(f"  {dict(zip(cols, row))}")


def load_existing_body_ids():
    """Load body IDs from the existing cx_heading_v1.json."""
    print(f"Loading existing circuit from {CX_FILE}...")
    with open(CX_FILE, 'r') as f:
        data = json.load(f)
    body_ids = [n["body_id"] for n in data["neurons"]]
    print(f"  Found {len(body_ids)} existing neurons")
    return body_ids


def batch_query(query_template, body_ids, batch_size=50):
    """Batch query to avoid parameter limits."""
    all_results = []
    total_batches = (len(body_ids) + batch_size - 1) // batch_size

    for i in range(0, len(body_ids), batch_size):
        batch = body_ids[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} IDs)...")

        ids_str = ", ".join(str(b) for b in batch)
        query = query_template.format(ids=ids_str)

        try:
            r = cypher(query)
            all_results.extend(r["data"])
        except Exception as e:
            print(f"    WARNING: Batch {batch_num} failed: {e}")

        time.sleep(0.1)  # Rate limiting

    return all_results


def findafferent_neurons(body_ids):
    """
    Find neurons that send connections TO the existing circuit (afferent).
    """
    print("\n[1/4] Finding afferent neurons (sending TO circuit)...")
    query_template = """
    MATCH (n:`hemibrain_Neuron`)-[c:ConnectsTo]->(m:`hemibrain_Neuron`)
    WHERE n.bodyId IN [{ids}]
      AND NOT m.bodyId IN [{ids}]
    RETURN DISTINCT
           m.bodyId AS bodyId,
           m.type AS type,
           m.instance AS instance,
           m.pre AS pre,
           m.post AS post,
           count(c) AS num_connections_to_circuit
    ORDER BY num_connections_to_circuit DESC
    """
    results = batch_query(query_template, body_ids)
    print(f"  Found {len(results)} afferent neurons")
    return results


def find_efferent_neurons(body_ids):
    """
    Find neurons that receive connections FROM the existing circuit (efferent).
    """
    print("\n[2/4] Finding efferent neurons (receiving FROM circuit)...")
    query_template = """
    MATCH (n:`hemibrain_Neuron`)-[c:ConnectsTo]->(m:`hemibrain_Neuron`)
    WHERE n.bodyId IN [{ids}]
      AND NOT m.bodyId IN [{ids}]
    RETURN DISTINCT
           n.bodyId AS bodyId,
           n.type AS type,
           n.instance AS instance,
           n.pre AS pre,
           n.post AS post,
           count(c) AS num_connections_to_target
    ORDER BY num_connections_to_target DESC
    """
    results = batch_query(query_template, body_ids)
    print(f"  Found {len(results)} efferent neurons")
    return results


def find_candidate_neurons(body_ids):
    """
    Find neurons connected to the circuit that are biologically relevant.
    """
    print("\n[3/4] Finding biologically relevant connected neurons...")
    types_str = ", ".join(f"'{t}'" for t in RELEVANT_TYPES)

    query_template = f"""
    MATCH (n:`hemibrain_Neuron`)
    WHERE n.type IN [{types_str}]
      AND (
        (n)-[:ConnectsTo]->(m:`hemibrain_Neuron`) WHERE m.bodyId IN [{{ids}}]
        OR
        (m:`hemibrain_Neuron`)-[:ConnectsTo]->(n) WHERE m.bodyId IN [{{ids}}]
      )
    RETURN DISTINCT
           n.bodyId AS bodyId,
           n.type AS type,
           n.instance AS instance,
           n.pre AS pre,
           n.post AS post
    ORDER BY n.type, n.instance
    """
    results = batch_query(query_template, body_ids)
    print(f"  Found {len(results)} relevant candidate neurons")
    return results


def find_circuit_connections(body_ids, candidate_ids):
    """
    Find connections between existing circuit and candidate neurons.
    """
    print("\n[4/4] Finding connections between circuit and candidates...")
    ids_str = ", ".join(str(b) for b in body_ids)
    cand_str = ", ".join(str(b) for b in candidate_ids)

    query = f"""
    MATCH (a:`hemibrain_Neuron`)-[c:ConnectsTo]->(b:`hemibrain_Neuron`)
    WHERE (
      (a.bodyId IN [{ids_str}] AND b.bodyId IN [{cand_str}])
      OR
      (a.bodyId IN [{cand_str}] AND b.bodyId IN [{ids_str}])
    )
    RETURN a.bodyId AS source_id,
           b.bodyId AS target_id,
           c.weight AS weight,
           a.type AS source_type,
           b.type AS target_type,
           CASE
             WHEN a.bodyId IN [{ids_str}] THEN 'efferent'
             ELSE 'afferent'
           END AS direction
    ORDER BY c.weight DESC
    """
    r = cypher(query)
    cols = r["columns"]
    results = [dict(zip(cols, row)) for row in r["data"]]
    print(f"  Found {len(results)} connections")
    return results


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing body IDs
    body_ids = load_existing_body_ids()

    # Find all connected neurons
    afferent = findafferent_neurons(body_ids)
    efferent = find_efferent_neurons(body_ids)

    # Combine and deduplicate
    all_candidates = {}
    for neuron in afferent + efferent:
        bid = neuron[0]
        if bid not in all_candidates:
            all_candidates[bid] = {
                "body_id": bid,
                "type": neuron[1],
                "instance": neuron[2],
                "pre": neuron[3],
                "post": neuron[4],
            }

    print(f"\nTotal unique connected neurons: {len(all_candidates)}")

    # Filter for relevant types
    relevant = []
    for bid, info in all_candidates.items():
        if info["type"] in RELEVANT_TYPES:
            relevant.append(info)

    print(f"Relevant connected neurons: {len(relevant)}")

    # Get candidate IDs
    candidate_ids = [n["body_id"] for n in relevant]

    # Find connections between circuit and candidates
    connections = find_circuit_connections(body_ids, candidate_ids)

    # Prepare output
    output = {
        "existing_circuit_size": len(body_ids),
        "afferent_neurons_found": len(afferent),
        "efferent_neurons_found": len(efferent),
        "total_unique_connected": len(all_candidates),
        "relevant_candidates": len(relevant),
        "candidate_neurons": relevant,
        "connections": connections,
        "target_size_range": "400-600",
        "current_expansion": len(relevant),
        "status": "audit_complete"
    }

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"\n=== SUMMARY ===")
    print(f"Existing circuit: {len(body_ids)} neurons")
    print(f"Afferent neurons: {len(afferent)}")
    print(f"Efferent neurons: {len(efferent)}")
    print(f"Total connected: {len(all_candidates)}")
    print(f"Relevant candidates: {len(relevant)}")
    print(f"Target range: 400-600")
    print(f"Expansion available: {len(relevant)} neurons")

    return output


if __name__ == "__main__":
    main()
