from flask import Flask
from src.app.routes.ai_routes import ai_routes
# from ai_routes import ai_routes

app = Flask(__name__)

app.register_blueprint(ai_routes, url_prefix='/ai')

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)