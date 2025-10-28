#!/bin/bash

# --- Script Configuration ---
# Stop script on any error
set -e

# --- User Variables (CHANGE THESE) ---
APP_USER="sillu"                 # System username to create for running the app
APP_DIR_NAME="stock_predictor_app"  # Directory name for the application
REPO_URL="https://github.com/silvadamarga/masinennustaja"  # *** REQUIRED: Replace with your Git repo URL ***
SERVER_NAME="_"                     # Nginx server_name (use "_" for IP only, or "yourdomain.com www.yourdomain.com")
WSGI_ENTRY_POINT="app:app"          # WSGI entry point (format: filename:flask_instance_name)
PYTHON_VERSION="python3.12"         # Adjust if you need a specific Python version supported by apt

# --- Derived Variables ---
APP_HOME="/home/${APP_USER}"
APP_FULL_PATH="${APP_HOME}/${APP_DIR_NAME}"
VENV_PATH="${APP_FULL_PATH}/venv"
SOCKET_FILE="${APP_FULL_PATH}/${APP_DIR_NAME}.sock"
SERVICE_NAME="flaskapp" # Name for the systemd service

# --- Helper Functions ---
print_info() {
    echo "--------------------------------------------------"
    echo "$1"
    echo "--------------------------------------------------"
}

# --- 1. Initial Server Setup ---
print_info "Updating system packages..."
sudo apt update
sudo apt upgrade -y

print_info "Creating application user '${APP_USER}'..."
if id "${APP_USER}" &>/dev/null; then
    echo "User ${APP_USER} already exists."
else
    sudo adduser --disabled-password --gecos "" "${APP_USER}"
    echo "User ${APP_USER} created."
fi
# Grant sudo if needed (optional, depends on your app's needs beyond running)
# sudo usermod -aG sudo "${APP_USER}"

# --- 2. Install Dependencies ---
print_info "Installing system dependencies (Python, Nginx, Git, Build Tools)..."
sudo apt install -y ${PYTHON_VERSION} ${PYTHON_VERSION}-pip ${PYTHON_VERSION}-venv python3-dev \
                   build-essential libssl-dev libffi-dev pkg-config git nginx curl

# --- 3. Firewall Setup ---
print_info "Configuring Firewall (UFW)..."
sudo ufw allow OpenSSH
sudo ufw allow http  # Port 80
sudo ufw allow https # Port 443
# Enable UFW non-interactively if it's not already enabled
if ! sudo ufw status | grep -q "Status: active"; then
    sudo ufw --force enable
fi
sudo ufw status

# --- 4. Deploy Application Code ---
print_info "Cloning application from ${REPO_URL}..."
# Use sudo -u to run git clone as the app user
if [ -d "${APP_FULL_PATH}" ]; then
    echo "App directory ${APP_FULL_PATH} already exists. Skipping clone."
    # Optional: Add logic here to pull latest changes if needed
    # sudo -u "${APP_USER}" git -C "${APP_FULL_PATH}" pull
else
    sudo -u "${APP_USER}" git clone "${REPO_URL}" "${APP_FULL_PATH}"
fi

# --- 5. Set Up Python Virtual Environment ---
print_info "Setting up Python virtual environment..."
# Run venv creation as the app user
sudo -u "${APP_USER}" ${PYTHON_VERSION} -m venv "${VENV_PATH}"

# --- 6. Install Python Packages ---
print_info "Installing Python packages from requirements.txt..."
# Activate venv and install within the app user's context
sudo -u "${APP_USER}" bash -c "source ${VENV_PATH}/bin/activate && pip install --upgrade pip && pip install wheel && pip install gunicorn && pip install -r ${APP_FULL_PATH}/requirements.txt"

# --- 7. Configure systemd Service ---
print_info "Creating systemd service file..."
sudo bash -c "cat > /etc/systemd/system/${SERVICE_NAME}.service" << EOL
[Unit]
Description=Gunicorn instance for ${APP_DIR_NAME}
After=network.target

[Service]
User=${APP_USER}
Group=www-data
WorkingDirectory=${APP_FULL_PATH}
Environment="PATH=${VENV_PATH}/bin"
ExecStart=${VENV_PATH}/bin/gunicorn --workers 3 --bind unix:${SOCKET_FILE} -m 007 ${WSGI_ENTRY_POINT}
Restart=always

[Install]
WantedBy=multi-user.target
EOL

print_info "Starting and enabling systemd service..."
sudo systemctl daemon-reload
sudo systemctl start "${SERVICE_NAME}"
sudo systemctl enable "${SERVICE_NAME}"
# Give it a moment to start up before checking status
sleep 2
sudo systemctl status "${SERVICE_NAME}" --no-pager # Check status (non-interactively)

# --- 8. Configure Nginx ---
print_info "Configuring Nginx..."
# Remove default site if it exists
sudo rm -f /etc/nginx/sites-enabled/default

sudo bash -c "cat > /etc/nginx/sites-available/${APP_DIR_NAME}" << EOL
server {
    listen 80;
    server_name ${SERVER_NAME}; # Use variable defined at the top

    location / {
        include proxy_params;
        proxy_pass http://unix:${SOCKET_FILE}; # Use the socket file
    }

    # Add other configurations like static file serving or SSL (Certbot) later
}
EOL

print_info "Enabling Nginx site and restarting Nginx..."
if [ ! -L "/etc/nginx/sites-enabled/${APP_DIR_NAME}" ]; then
    sudo ln -s "/etc/nginx/sites-available/${APP_DIR_NAME}" "/etc/nginx/sites-enabled/"
fi
sudo nginx -t # Test configuration
sudo systemctl restart nginx

# --- 9. Final Instructions ---
print_info "Deployment Script Finished!"
echo ""
echo "Your Flask application should now be accessible via Nginx."
echo "If you used an IP address, visit: http://${SERVER_NAME}"
echo "If you used a domain name, visit: http://yourdomain.com"
echo ""
echo "Next Steps:"
echo "1. If using a domain, configure DNS A records to point to your server's IP."
echo "2. Set up HTTPS using Certbot (recommended):"
echo "   sudo apt install certbot python3-certbot-nginx -y"
echo "   sudo certbot --nginx # Follow prompts"
echo "3. Check service status: sudo systemctl status ${SERVICE_NAME}"
echo "4. Check Nginx status: sudo systemctl status nginx"
echo "5. View Gunicorn logs: sudo journalctl -u ${SERVICE_NAME}"
echo "6. View Nginx logs: tail -f /var/log/nginx/access.log /var/log/nginx/error.log"
echo ""
echo "Remember to manage sensitive environment variables (e.g., API keys) securely,"
echo "either via a .env file loaded by your app (ensure it's gitignored!) or"
echo "by adding Environment= lines to the systemd service file."
echo ""
