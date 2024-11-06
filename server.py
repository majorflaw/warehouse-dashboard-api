from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import asyncio
from typing import List
from datetime import datetime
import logging
import os
from dropbox import Dropbox
from dropbox.files import FileMetadata
from io import BytesIO
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_modified = {}  # Track last modified time for each file
        self.last_data = None
        self.monitoring = False
        # Initialize Dropbox client
        self.dbx = Dropbox(os.getenv('DROPBOX_ACCESS_TOKEN'))
        
        # Define file paths in Dropbox
        self.files = {
            'shipments': '/open_shipments.csv',
            'p2b_stats': '/p2b_statistics.json',
            'legacy_stats': '/legacy_statistics.json',
            'total_stats': '/total_statistics.json'
        }

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast_data(self, data: dict):
        logger.info(f"Broadcasting to {len(self.active_connections)} clients")
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
                logger.info("Successfully sent data to a client")
            except Exception as e:
                logger.error(f"Error sending data: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def read_file_from_dropbox(self, path: str):
        """Read file content from Dropbox"""
        try:
            metadata, response = self.dbx.files_download(path)
            content = response.content
            
            if path.endswith('.csv'):
                return pd.read_csv(BytesIO(content))
            elif path.endswith('.json'):
                return json.loads(content.decode())
            
        except Exception as e:
            logger.error(f"Error reading file from Dropbox: {e}")
            return None

    async def check_file_changes(self):
        """Check if any files have changed in Dropbox"""
        try:
            changes_detected = False
            
            for file_key, file_path in self.files.items():
                try:
                    metadata = self.dbx.files_get_metadata(file_path)
                    last_modified = metadata.server_modified
                    
                    if (file_key not in self.last_modified or 
                        self.last_modified[file_key] != last_modified):
                        logger.info(f"Change detected in {file_key}")
                        self.last_modified[file_key] = last_modified
                        changes_detected = True
                except Exception as e:
                    logger.error(f"Error checking {file_key}: {e}")
            
            return changes_detected
            
        except Exception as e:
            logger.error(f"Error checking file changes: {e}")
            return False

    async def read_all_data(self):
        """Read all necessary data from Dropbox"""
        try:
            # Read shipments data
            df = await self.read_file_from_dropbox(self.files['shipments'])
            if df is None:
                return None
                
            df = df.replace([np.inf, -np.inf], None)
            df = df.replace({np.nan: None})
            return df.to_dict(orient='records')
            
        except Exception as e:
            logger.error(f"Error reading all data: {e}")
            return None

manager = ConnectionManager()

async def monitor_file_changes():
    """Monitor for file changes and broadcast updates"""
    logger.info("Starting file monitoring task")
    manager.monitoring = True
    
    try:
        while manager.monitoring:
            try:
                if await manager.check_file_changes():
                    data = await manager.read_all_data()
                    if data is not None:
                        # Convert current data to JSON for comparison
                        current_data_json = json.dumps(data, sort_keys=True)
                        last_data_json = json.dumps(manager.last_data, sort_keys=True) if manager.last_data else None
                        
                        if current_data_json != last_data_json:
                            logger.info("Data content changed, broadcasting update")
                            await manager.broadcast_data({
                                "type": "data_update",
                                "data": data,
                                "timestamp": datetime.now().isoformat()
                            })
                            manager.last_data = data
                        else:
                            logger.debug("Data content unchanged")
                    else:
                        logger.warning("Failed to read data")
                
                # Reduced polling interval for more frequent updates
                await asyncio.sleep(30)  # Check every 30 seconds instead of 60
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
            
    except Exception as e:
        logger.error(f"Fatal error in monitor task: {e}")
        manager.monitoring = False
    
    logger.info("File monitoring task stopped")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create the monitoring task
    logger.info("Starting up server and monitoring task")
    monitoring_task = asyncio.create_task(monitor_file_changes())
    yield
    # Shutdown: Cancel the monitoring task
    logger.info("Shutting down server and monitoring task")
    manager.monitoring = False
    monitoring_task.cancel()
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    logger.info("Server shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Warehouse Dashboard API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://majorflaw.github.io",
        "http://localhost:5173",  # Keep local development URL
        "http://localhost:3000"   # Common React development URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Send initial data
        data = await manager.read_all_data()
        if data is not None:
            await websocket.send_json({
                "type": "initial_data",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            logger.info("Sent initial data to new client")

        # Keep connection alive and handle messages
        while True:
            try:
                message = await websocket.receive_text()
                if message == "ping":
                    await websocket.send_text("pong")
                    logger.debug("Received ping, sent pong")
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                break
            except Exception as e:
                logger.error(f"Error handling websocket message: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy"}