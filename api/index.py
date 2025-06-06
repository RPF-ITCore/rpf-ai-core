from mangum import Mangum
from main import app  # assuming 'app' is created in main.py

handler = Mangum(app)
