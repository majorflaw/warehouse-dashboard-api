from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import asyncio
from typing import List, Optional, Tuple
from datetime import datetime
import logging
import os
import time
from dropbox import Dropbox
from dropbox.oauth import DropboxOAuth2FlowNoRedirect
from dropbox.exceptions import AuthError, ApiError
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

class DropboxClient:
    def __init__(self, app_key: str, app_secret: str, refresh_token: str):
        """
        Initialize the DropboxClient with OAuth2 credentials.
        
        Args:
            app_key (str): Dropbox app key
            app_secret (str): Dropbox app secret
            refresh_token (str): OAuth2 refresh token
        """
        if not app_key or not app_secret or not refresh_token:
            raise ValueError("All OAuth2 credentials (app_key, app_secret, refresh_token) must be provided")
        
        self.app_key = app_key
        self.app_secret = app_secret
        self.refresh_token = refresh_token
        self._client: Optional[Dropbox] = None
        
        # Log initialization but not the sensitive values
        logger.info("Initializing DropboxClient")
        logger.debug("App key length: %d", len(app_key))
        logger.debug("App secret length: %d", len(app_secret))
        logger.debug("Refresh token length: %d", len(refresh_token))

    async def get_client(self) -> Dropbox:
        """Get a Dropbox client, creating it if necessary."""
        if not self._client:
            await self._create_client()
        return self._client

    async def _create_client(self):
        """Create a new Dropbox client instance."""
        try:
            self._client = Dropbox(
                oauth2_refresh_token=self.refresh_token,
                app_key=self.app_key,
                app_secret=self.app_secret,
                session=None
            )
            logger.info("Successfully created new Dropbox client")
        except Exception as e:
            logger.error(f"Error creating Dropbox client: {str(e)}")
            raise

    async def files_download(self, path: str) -> Tuple[FileMetadata, bytes]:
        """
        Download a file from Dropbox with automatic retry logic.
        
        Args:
            path (str): Path to the file in Dropbox
            
        Returns:
            Tuple[FileMetadata, bytes]: File metadata and content
            
        Raises:
            ApiError: If there's an error accessing Dropbox API
            AuthError: If there's an authentication error
            Exception: For other errors
        """
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                client = await self.get_client()
                metadata, response = client.files_download(path)
                logger.info(f"Successfully downloaded file: {path}")
                return metadata, response

            except AuthError as e:
                logger.warning(f"Authentication error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    await self._create_client()
                    time.sleep(retry_delay)
                else:
                    raise

            except ApiError as e:
                if e.error.is_path() and e.error.get_path().is_not_found():
                    logger.error(f"File not found: {path}")
                    raise
                logger.error(f"Dropbox API error: {str(e)}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error downloading file: {str(e)}")
                raise

    async def files_get_metadata(self, path: str) -> FileMetadata:
        """
        Get metadata for a file in Dropbox with automatic retry logic.
        
        Args:
            path (str): Path to the file in Dropbox
            
        Returns:
            FileMetadata: Metadata of the file
            
        Raises:
            ApiError: If there's an error accessing Dropbox API
            AuthError: If there's an authentication error
            Exception: For other errors
        """
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                client = await self.get_client()
                metadata = client.files_get_metadata(path)
                logger.info(f"Successfully got metadata for: {path}")
                return metadata

            except AuthError as e:
                logger.warning(f"Authentication error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    await self._create_client()
                    time.sleep(retry_delay)
                else:
                    raise

            except ApiError as e:
                if e.error.is_path() and e.error.get_path().is_not_found():
                    logger.error(f"File not found: {path}")
                    raise
                logger.error(f"Dropbox API error: {str(e)}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error getting metadata: {str(e)}")
                raise

    async def read_file(self, path: str) -> Optional[pd.DataFrame | dict]:
        """
        Read a file from Dropbox and return its contents as either a DataFrame or dict.
        
        Args:
            path (str): Path to the file in Dropbox
            
        Returns:
            Optional[Union[pd.DataFrame, dict]]: File contents as DataFrame for CSV files,
                                               dict for JSON files, or None on error
        """
        try:
            _, content = await self.files_download(path)
            
            if path.endswith('.csv'):
                return pd.read_csv(BytesIO(content))
            elif path.endswith('.json'):
                return json.loads(content.decode())
            else:
                logger.error(f"Unsupported file type: {path}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            return None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_modified = {}
        self.last_data = None
        self.monitoring = False
        
        # Initialize Dropbox client
        try:
            self.dbx = DropboxClient(
                app_key=os.getenv('DROPBOX_APP_KEY', ''),
                app_secret=os.getenv('DROPBOX_APP_SECRET', ''),
                refresh_token=os.getenv('DROPBOX_REFRESH_TOKEN', '')
            )
        except Exception as e:
            logger.error(f"Failed to initialize DropboxClient: {e}")
            raise

        # Define file paths in Dropbox
        self.files = {
            'shipments': '/warehouse-dashboard/open_shipments.csv',
            'p2b_stats': '/warehouse-dashboard/p2b_statistics.json',
            'legacy_stats': '/warehouse-dashboard/legacy_statistics.json',
            'total_stats': '/warehouse-dashboard/total_statistics.json'
        }

    # WebSocket connection management methods
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

    # Dropbox interaction methods
    async def read_file_from_dropbox(self, path: str):
        """Read file content from Dropbox"""
        try:
            metadata, response = await self.dbx.files_download(path)
            
            if path.endswith('.csv'):
                return pd.read_csv(BytesIO(response.content))
            elif path.endswith('.json'):
                return json.loads(response.content.decode())
            
        except Exception as e:
            logger.error(f"Error reading file from Dropbox: {e}")
            return None

    async def check_file_changes(self):
        """Check if any files have changed in Dropbox"""
        try:
            changes_detected = False
            
            for file_key, file_path in self.files.items():
                try:
                    metadata = await self.dbx.files_get_metadata(file_path)
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
            shipments_df = await self.read_file_from_dropbox(self.files['shipments'])
            if shipments_df is None:
                return None, None
                
            shipments_df = shipments_df.replace([np.inf, -np.inf], None)
            shipments_df = shipments_df.replace({np.nan: None})
            shipments_data = shipments_df.to_dict(orient='records')

            # Read statistics data
            p2b_stats = await self.read_file_from_dropbox(self.files['p2b_stats'])
            legacy_stats = await self.read_file_from_dropbox(self.files['legacy_stats'])
            total_stats = await self.read_file_from_dropbox(self.files['total_stats'])

            statistics = {
                'p2b': p2b_stats,
                'legacy': legacy_stats,
                'total': total_stats
            }

            return shipments_data, statistics
                
        except Exception as e:
            logger.error(f"Error reading all data: {e}")
            return None, None

manager = ConnectionManager()

async def monitor_file_changes():
    """Monitor for file changes and broadcast updates"""
    logger.info("Starting file monitoring task")
    manager.monitoring = True
    
    try:
        while manager.monitoring:
            try:
                if await manager.check_file_changes():
                    data, statistics = await manager.read_all_data()
                    if data is not None:
                        # Convert current data to JSON for comparison
                        current_data_json = json.dumps(data, sort_keys=True)
                        last_data_json = json.dumps(manager.last_data, sort_keys=True) if manager.last_data else None
                        
                        if current_data_json != last_data_json:
                            logger.info("Data content changed, broadcasting update")
                            await manager.broadcast_data({
                                "type": "data_update",
                                "data": data,
                                "statistics": statistics,
                                "timestamp": datetime.now().isoformat()
                            })
                            manager.last_data = data
                        else:
                            logger.debug("Data content unchanged")
                    else:
                        logger.warning("Failed to read data")
                
                # Reduced polling interval for more frequent updates
                await asyncio.sleep(30)  # Check every 30 seconds
                
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
        data, statistics = await manager.read_all_data()
        if data is not None:
            await websocket.send_json({
                "type": "initial_data",
                "data": data,
                "statistics": statistics,
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
