from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import logging
from datetime import datetime
from modules.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'crypto-analytics-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

db = DatabaseManager()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'WebSocket Server'}), 200

@app.route('/api/tick', methods=['POST'])
def receive_tick():
    try:
        data = request.get_json()
        
        required_fields = ['symbol', 'price', 'size']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())
        
        db.insert_tick(
            symbol=str(data['symbol']).upper(),
            timestamp=timestamp,
            price=float(data['price']),
            size=float(data['size']),
            source='external_api'
        )
        
        socketio.emit('tick_received', {
            'symbol': data['symbol'],
            'price': data['price'],
            'size': data['size'],
            'timestamp': timestamp
        }, namespace='/')
        
        return jsonify({'status': 'success', 'message': 'Tick data received'}), 200
    
    except Exception as e:
        logger.error(f"Error receiving tick data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ticks/batch', methods=['POST'])
def receive_ticks_batch():
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({'error': 'Expected array of tick data'}), 400
        
        ticks = []
        for tick in data:
            required_fields = ['symbol', 'price', 'size']
            for field in required_fields:
                if field not in tick:
                    continue
            
            timestamp = tick.get('timestamp') or tick.get('ts', datetime.utcnow().isoformat())
            
            ticks.append({
                'symbol': str(tick['symbol']).upper(),
                'ts': timestamp,
                'price': float(tick['price']),
                'size': float(tick['size']),
                'source': 'external_api'
            })
        
        if ticks:
            db.insert_ticks_batch(ticks)
            
            socketio.emit('ticks_batch_received', {
                'count': len(ticks),
                'timestamp': datetime.utcnow().isoformat()
            }, namespace='/')
        
        return jsonify({'status': 'success', 'message': f'{len(ticks)} ticks received'}), 200
    
    except Exception as e:
        logger.error(f"Error receiving tick batch: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected', 'client_id': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('send_tick')
def handle_send_tick(data):
    try:
        required_fields = ['symbol', 'price', 'size']
        for field in required_fields:
            if field not in data:
                emit('error', {'message': f'Missing required field: {field}'})
                return
        
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())
        
        db.insert_tick(
            symbol=str(data['symbol']).upper(),
            timestamp=timestamp,
            price=float(data['price']),
            size=float(data['size']),
            source='websocket_client'
        )
        
        emit('tick_acknowledged', {
            'symbol': data['symbol'],
            'timestamp': timestamp,
            'status': 'saved'
        }, broadcast=False)
        
        emit('tick_received', data, broadcast=True, include_self=False)
        
        logger.info(f"Tick received via WebSocket: {data['symbol']} @ {data['price']}")
    
    except Exception as e:
        logger.error(f"Error processing WebSocket tick: {e}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    logger.info("Starting WebSocket Server on port 5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)
