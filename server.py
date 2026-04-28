"""
Combined HTTP + WebSocket Server for Agent-Based Simulation
(Streaming Architecture)

- HTTP (Flask) on port 5000: Commands (init, start, stop, step, stats)
- WebSocket on port 8765: Streaming updates at constant rate

Server is simulation-agnostic:
- Does not know about agent types or topology details
- Passes config data through to simulation without inspection
- Position coordinates are opaque (could be 2D, 3D, graph nodes, etc.)

Usage:
    python combined_server.py
"""

import asyncio
import threading
import sys
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import streaming server components
import streaming_server as ws

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins


# ============================================================
# Health Check
# ============================================================

@app.route('/', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'agent-simulation'})


@app.route('/health', methods=['GET'])
def health():
    """Health check with more details"""
    return jsonify({
        'status': 'ok',
        'world_initialized': ws.world is not None,
        'simulation_running': ws.simulation_running
    })


# ============================================================
# HTTP Endpoints (Commands & Queries)
# ============================================================

@app.route('/init', methods=['POST'])
def init_world():
    data = request.get_json(silent=True) or {}
    result = ws.handle_http_init(data)
    return jsonify(result)


@app.route('/start', methods=['POST'])
def start_simulation():
    data = request.get_json(silent=True) or {}
    result = ws.handle_http_start(data)
    return jsonify(result)


@app.route('/stop', methods=['POST'])
def stop_simulation():
    data = request.get_json(silent=True) or {}
    result = ws.handle_http_stop(data)
    return jsonify(result)


@app.route('/step', methods=['POST'])
def step_simulation():
    data = request.get_json(silent=True) or {}
    result = ws.handle_http_step(data)
    return jsonify(result)


@app.route('/stats', methods=['GET'])
def get_stats():
    result = ws.handle_http_stats()
    return jsonify(result)


def parse_position(pos_str: str, default=None):
    """
    Parse an underscore-separated position string into a list of floats.
    
    Examples:
        "50_50" -> [50.0, 50.0]
        "10_20_30" -> [10.0, 20.0, 30.0]
        "node_5" -> Would fail (non-numeric) - handle in simulation
    
    For non-numeric positions (e.g., graph nodes), the simulation
    would need to define its own encoding scheme.
    """
    if pos_str is None:
        return default
    try:
        return [float(x) for x in pos_str.split('_')]
    except ValueError:
        return None


@app.route('/viewport', methods=['GET'])
def get_viewport():
    """
    Get current state of agents in viewport.
    
    Query params:
        center: Underscore-separated position coordinates (e.g., "50_50" for 2D)
        r: radius
    
    Examples:
        /viewport?center=50_50&r=20          (2D)
        /viewport?center=10_20_30&r=15       (3D)
    
    For backward compatibility, also accepts cx/cy format:
        /viewport?cx=50&cy=50&r=20
    
    This is the "active view" endpoint for real-time viewport updates.
    Client polls at desired rate - always returns fresh data.
    """
    try:
        radius = float(request.args.get('r', 20))
        
        # Try new format first: center=x_y_z...
        center_str = request.args.get('center')
        if center_str:
            center = parse_position(center_str)
            if center is None:
                return jsonify({'error': 'Invalid center format', 'agents': [], 'count': 0}), 400
        else:
            # Fall back to old format: cx=x&cy=y (for backward compatibility)
            cx = request.args.get('cx')
            cy = request.args.get('cy')
            if cx is not None and cy is not None:
                center = [float(cx), float(cy)]
            else:
                # Default center
                center = [50.0, 50.0]
                
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid parameters: {e}', 'agents': [], 'count': 0}), 400
    
    # Pass center as opaque position to handler
    result = ws.handle_http_viewport(center, radius)
    return jsonify(result)


@app.route('/inspect/<int:agent_id>', methods=['GET'])
def inspect_agent(agent_id):
    """
    Get full state of a specific agent.
    
    Example: /inspect/12345
    
    Returns complete internal state (not just phenotype).
    """
    result = ws.handle_http_inspect(agent_id)
    return jsonify(result)


@app.route('/params', methods=['GET'])
def get_params():
    """Get simulation parameters - delegates to World.report_params()"""
    from sim import World
    return jsonify(World.report_params())


@app.route('/set_param', methods=['POST'])
def set_param():
    """
    Set a simulation parameter.
    
    Body:
        {"name": "prey.move_cost", "value": 0.2}
    
    Returns:
        {"status": "ok", "name": "...", "old_value": ..., "new_value": ...}
        {"status": "deferred", "name": "...", "message": "..."}
        {"status": "error", "message": "..."}
    """
    from sim import World
    
    data = request.get_json(silent=True) or {}
    name = data.get('name')
    value = data.get('value')
    
    if name is None:
        return jsonify({'status': 'error', 'message': 'Missing parameter name'}), 400
    if value is None:
        return jsonify({'status': 'error', 'message': 'Missing parameter value'}), 400

    if ws.world is None:
        return jsonify({'status': 'error', 'message': 'World is not initialized'}), 400
    
    result = ws.world.set_param(name, value)
    
    if result.get('status') == 'error':
        return jsonify(result), 400
    
    return jsonify(result)


@app.route('/streaming/config', methods=['GET'])
def get_streaming_config():
    """Get streaming configuration"""
    return jsonify({
        'default_interval_ms': ws.DEFAULT_STREAM_INTERVAL_MS,
        'default_entities_per_message': ws.DEFAULT_ENTITIES_PER_MESSAGE,
        'max_despawns_per_message': ws.MAX_DESPAWNS_PER_MESSAGE,
        'min_interval_ms': ws.MIN_STREAM_INTERVAL_MS,
        'max_interval_ms': ws.MAX_STREAM_INTERVAL_MS,
        'min_chunk_size': ws.MIN_ENTITIES_PER_MESSAGE,
        'max_chunk_size': ws.MAX_ENTITIES_PER_MESSAGE,
        'theoretical_max_entities_per_sec': 1000 / ws.MIN_STREAM_INTERVAL_MS * ws.MAX_ENTITIES_PER_MESSAGE
    })


# ============================================================
# Server Startup
# ============================================================

def run_flask():
    """Run Flask HTTP server in a thread"""
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


def run_websocket():
    """Run WebSocket server in async event loop"""
    asyncio.run(ws.main(host='0.0.0.0', port=8765))


if __name__ == '__main__':
    # Ensure print statements are immediately visible
    sys.stdout.reconfigure(line_buffering=True)
    
    print("=" * 60)
    print("Agent-Based Simulation Server (Streaming Architecture)")
    print("=" * 60)
    print()
    print("HTTP Endpoints (port 5000):")
    print("  POST /init         - Initialize world")
    print("  POST /start        - Start simulation")
    print("  POST /stop         - Stop simulation")
    print("  POST /step         - Single step(s)")
    print("  POST /set_param    - Set simulation parameter")
#    print("  GET  /full        - Get full world state")
    print("  GET  /stats        - Get statistics")
    print("  GET  /params       - Get parameters")
    print("  GET  /viewport     - Get agents in viewport (center, r)")
    print("  GET  /inspect/<id> - Get full agent state")
    print()
    print("Viewport URL format:")
    print("  /viewport?center=50_50&r=20       (new format)")
    print("  /viewport?cx=50&cy=50&r=20        (legacy format)")
    print()
    print("WebSocket (port 8765):")
    print("  -> subscribe      - Start receiving streaming updates")
    print("  -> unsubscribe    - Stop receiving updates")
    print("  -> viewport       - Set viewport filter for WS stream")
    print("  -> viewport_clear - Clear viewport filter")
    print("  <- snapshot       - Full state (on subscribe)")
    print("  <- delta          - Streaming updates (constant rate)")
    print("  <- stats          - Periodic statistics")
    print()
    print("Recommended usage:")
    print("  - WebSocket: Background world updates (slow, complete)")
    print("  - HTTP /viewport: Active view polling (fast, fresh)")
    print()
    print(f"WS Streaming: {ws.DEFAULT_ENTITIES_PER_MESSAGE} agents every {ws.DEFAULT_STREAM_INTERVAL_MS}ms")
    print(f"              = {1000/ws.DEFAULT_STREAM_INTERVAL_MS * ws.DEFAULT_ENTITIES_PER_MESSAGE:.0f} agent updates/sec")
    print()
    print("=" * 60)
    
    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("[HTTP] Server started on http://0.0.0.0:5000")
    
    # Run WebSocket in main thread (needs asyncio event loop)
    print("[WS] Server starting on ws://0.0.0.0:8765")
    run_websocket()
