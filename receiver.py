from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class SimpleReceiver(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = self.headers.get('Content-Length')
        if content_length is None:
            self.send_response(411)  # Length Required
            self.end_headers()
            self.wfile.write(b'{"status":"Error: No Content-Length header"}')
            return
        try:
            content_length = int(content_length)
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))
            print("📩 Received:", data)
            response = {"status": "Received", "data": data}
        except Exception as e:
            print("❌ Error parsing data:", e)
            response = {"status": "Error", "error": str(e)}
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

if __name__ == "__main__":
    server_address = ("0.0.0.0", 5000)
    httpd = HTTPServer(server_address, SimpleReceiver)
    print("🚀 Receiver running on port 5000...")
    httpd.serve_forever()