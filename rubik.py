# ------------------ rubik.py ------------------
import kociemba
from flask import Flask, request, jsonify, send_from_directory
import os
from collections import Counter
import serial
import serial.tools.list_ports
import time

# ------------------ Flask Web Server ------------------
flask_app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Color mapping
COLOR_MAP = {
    "white": "U",
    "red": "R", 
    "green": "F",
    "yellow": "D",
    "orange": "L",
    "blue": "B"
}

# Arduino communication class
class ArduinoController:
    def __init__(self):
        self.ser = None
        self.connected = False
        self.port = None
        self.baudrate = 9600
        
    def find_arduino_port(self):
        """Automatically find Arduino port"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'arduino' in port.description.lower() or 'CH340' in port.description or 'USB Serial' in port.description:
                return port.device
        return None
    
    def connect(self, port=None):
        """Connect to Arduino"""
        try:
            if port is None:
                port = self.find_arduino_port()
                if port is None:
                    return False, "Arduino not found. Please check connection."
            
            self.ser = serial.Serial(port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            self.connected = True
            self.port = port
            print(f"Connected to Arduino on {port}")
            return True, f"Connected to {port}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.connected = False
        self.ser = None
    
    def send_solution(self, solution):
        """Send solution to Arduino"""
        if not self.connected or not self.ser:
            return False, "Not connected to Arduino"
        
        try:
            # Convert solution to Arduino commands
            moves = solution.split()
            command_string = "SOLVE:" + ",".join(moves) + "\n"
            
            self.ser.write(command_string.encode())
            time.sleep(0.1)
            
            # Wait for acknowledgment
            ack = self.ser.readline().decode().strip()
            if ack == "ACK":
                return True, "Solution sent successfully"
            else:
                return False, f"No acknowledgment received: {ack}"
                
        except Exception as e:
            return False, f"Send failed: {str(e)}"
    
    def send_single_move(self, move):
        """Send a single move to Arduino"""
        if not self.connected or not self.ser:
            return False, "Not connected to Arduino"
        
        try:
            command_string = f"MOVE:{move}\n"
            self.ser.write(command_string.encode())
            time.sleep(0.1)
            
            # Wait for acknowledgment
            ack = self.ser.readline().decode().strip()
            if ack == "ACK":
                return True, f"Move {move} sent successfully"
            else:
                return False, f"No acknowledgment received: {ack}"
                
        except Exception as e:
            return False, f"Send failed: {str(e)}"
    
    def send_test_command(self):
        """Send test command to verify connection"""
        if not self.connected:
            return False, "Not connected"
        
        try:
            self.ser.write("TEST\n".encode())
            response = self.ser.readline().decode().strip()
            return True, f"Arduino response: {response}"
        except Exception as e:
            return False, f"Test failed: {str(e)}"

# Initialize Arduino controller
arduino = ArduinoController()

def validate_cube_state(cube_string):
    """Validate if the cube state is solvable"""
    if len(cube_string) != 54:
        return False, "Cube string must be exactly 54 characters"
    
    color_count = Counter(cube_string)
    expected_colors = {'U', 'R', 'F', 'D', 'L', 'B'}
    
    if set(color_count.keys()) != expected_colors:
        return False, "All six colors must be present"
    
    for color in expected_colors:
        if color_count[color] != 9:
            return False, f"Color {color} should appear exactly 9 times"
    
    return True, "Valid cube state"

@flask_app.route("/")
def index():
    return send_from_directory(BASE_DIR, "rubik.html")

@flask_app.route("/solve", methods=["POST"])
def solve_cube():
    try:
        data = request.json
        faces_data = data.get("faces", {})
        send_to_arduino = data.get("send_to_arduino", False)
        
        face_order = ["U", "R", "F", "D", "L", "B"]
        cube_string = ""
        
        for face in face_order:
            if face not in faces_data:
                return jsonify({"error": f"Missing {face} face data"})
            
            stickers = faces_data[face]
            if len(stickers) != 9:
                return jsonify({"error": f"{face} face does not have 9 stickers"})
            
            for color_name in stickers:
                if color_name in COLOR_MAP:
                    cube_string += COLOR_MAP[color_name]
                else:
                    return jsonify({"error": f"Unknown color '{color_name}' on {face} face"})
        
        is_valid, message = validate_cube_state(cube_string)
        if not is_valid:
            return jsonify({"error": message})
        
        try:
            solution = kociemba.solve(cube_string)
            move_count = len(solution.split())
            
            response_data = {
                "success": True,
                "cube_string": cube_string,
                "solution": solution,
                "move_count": move_count,
                "solver_used": "kociemba"
            }
            
            # Send to Arduino if requested
            if send_to_arduino and arduino.connected:
                arduino_success, arduino_message = arduino.send_solution(solution)
                response_data["arduino_sent"] = arduino_success
                response_data["arduino_message"] = arduino_message
            elif send_to_arduino and not arduino.connected:
                response_data["arduino_sent"] = False
                response_data["arduino_message"] = "Arduino not connected"
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({"error": f"Cannot solve this cube state: {str(e)}"})
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"})

@flask_app.route("/arduino/connect", methods=["POST"])
def connect_arduino():
    """Connect to Arduino"""
    data = request.json
    port = data.get("port", None)
    
    success, message = arduino.connect(port)
    return jsonify({"success": success, "message": message})

@flask_app.route("/arduino/disconnect", methods=["POST"])
def disconnect_arduino():
    """Disconnect from Arduino"""
    arduino.disconnect()
    return jsonify({"success": True, "message": "Disconnected"})

@flask_app.route("/arduino/status", methods=["GET"])
def arduino_status():
    """Get Arduino connection status"""
    return jsonify({
        "connected": arduino.connected,
        "port": arduino.port
    })

@flask_app.route("/arduino/test", methods=["POST"])
def test_arduino():
    """Test Arduino connection"""
    if not arduino.connected:
        return jsonify({"success": False, "message": "Arduino not connected"})
    
    success, message = arduino.send_test_command()
    return jsonify({"success": success, "message": message})

@flask_app.route("/arduino/send-move", methods=["POST"])
def send_single_move():
    """Send a single move to Arduino"""
    try:
        if not arduino.connected:
            return jsonify({"success": False, "message": "Arduino not connected"})
        
        data = request.json
        move = data.get('move', '')
        
        if not move:
            return jsonify({"success": False, "message": "No move provided"})
        
        success, message = arduino.send_single_move(move)
        return jsonify({"success": success, "message": message})
            
    except Exception as e:
        return jsonify({"success": False, "message": f"Send failed: {str(e)}"})

@flask_app.route("/arduino/ports", methods=["GET"])
def list_ports():
    """List available serial ports"""
    ports = [port.device for port in serial.tools.list_ports.comports()]
    return jsonify({"ports": ports})

@flask_app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "arduino_connected": arduino.connected})

def run_flask():
    flask_app.run(debug=False, use_reloader=False, port=5000, host='127.0.0.1')

if __name__ == "__main__":
    print("Starting Rubik's Cube Solver with Arduino Integration...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    
    # Try to auto-connect to Arduino
    print("Searching for Arduino...")
    success, message = arduino.connect()
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    
    run_flask()