"""MechDoctor backend entry point.

Run locally:   python app.py            (from the backend/ folder)
Production:    gunicorn app:app         (Render start command)
"""
import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS

import config
from routes.diagnose import bp as diagnose_bp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app():
    app = Flask(__name__)

    # FRONTEND_ORIGIN can hold several origins separated by commas, e.g.
    # "http://localhost:5173,https://mechdoctor.vercel.app"
    origins = [origin.strip() for origin in config.FRONTEND_ORIGIN.split(",") if origin.strip()]
    CORS(app, origins=origins)

    app.register_blueprint(diagnose_bp)

    @app.get("/api/health")
    def health():
        # Booleans only: shows whether keys are configured without ever exposing them.
        return jsonify({
            "status": "ok",
            "ai": {
                "gemini_configured": bool(config.GEMINI_API_KEY),
                "groq_configured": bool(config.GROQ_API_KEY),
            },
        })

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return jsonify({"error": "Method not allowed."}), 405

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )