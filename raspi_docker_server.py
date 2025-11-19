#!/usr/bin/env python3
"""
Raspberry Pi Docker API Server Manager
Sets up and manages the pedal-backend Docker container
"""

import subprocess
import sys
import time
import json
from pathlib import Path

class DockerServerManager:
    def __init__(self):
        self.container_name = "pedal-backend"
        self.image_name = "kaushiksannidhi/pedal-backend:latest"
        self.port = 8000
        self.host_port = 8000
        
    def run_command(self, cmd, check=True):
        """Execute shell command and return output"""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            return result.stdout.strip(), result.returncode
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {cmd}")
            print(f"Error: {e.stderr}")
            return e.stderr, e.returncode
    
    def check_docker_installed(self):
        """Check if Docker is installed"""
        print("Checking Docker installation...")
        output, code = self.run_command("docker --version", check=False)
        if code != 0:
            print("❌ Docker is not installed!")
            print("\nInstall Docker with:")
            print("curl -fsSL https://get.docker.com -o get-docker.sh")
            print("sudo sh get-docker.sh")
            print("sudo usermod -aG docker $USER")
            return False
        print(f"✓ Docker found: {output}")
        return True
    
    def stop_existing_container(self):
        """Stop and remove existing container if running"""
        print(f"\nChecking for existing {self.container_name} container...")
        output, _ = self.run_command(
            f"docker ps -a --filter name={self.container_name} --format '{{{{.Names}}}}'",
            check=False
        )
        
        if output:
            print(f"Found existing container, stopping and removing...")
            self.run_command(f"docker stop {self.container_name}", check=False)
            self.run_command(f"docker rm {self.container_name}", check=False)
            print("✓ Cleaned up old container")
    
    def pull_image(self):
        """Pull the latest Docker image"""
        print(f"\nPulling Docker image: {self.image_name}")
        print("This may take a few minutes...")
        output, code = self.run_command(f"docker pull {self.image_name}")
        if code != 0:
            print("❌ Failed to pull image")
            return False
        print("✓ Image pulled successfully")
        return True
    
    def start_container(self):
        """Start the Docker container"""
        print(f"\nStarting container on port {self.host_port}...")
        
        cmd = f"""docker run -d \
            --name {self.container_name} \
            --restart unless-stopped \
            -p {self.host_port}:{self.port} \
            {self.image_name}"""
        
        output, code = self.run_command(cmd)
        if code != 0:
            print("❌ Failed to start container")
            return False
        
        print(f"✓ Container started with ID: {output[:12]}")
        return True
    
    def get_local_ip(self):
        """Get local IP address"""
        output, _ = self.run_command(
            "hostname -I | awk '{print $1}'",
            check=False
        )
        return output if output else "localhost"
    
    def check_container_health(self):
        """Check if container is running and healthy"""
        print("\nChecking container health...")
        time.sleep(3)  # Wait for container to start
        
        output, code = self.run_command(
            f"docker ps --filter name={self.container_name} --format '{{{{.Status}}}}'",
            check=False
        )
        
        if code == 0 and output:
            print(f"✓ Container status: {output}")
            return True
        else:
            print("❌ Container is not running")
            # Show logs
            logs, _ = self.run_command(f"docker logs {self.container_name}", check=False)
            print("\nContainer logs:")
            print(logs)
            return False
    
    def display_info(self):
        """Display connection information"""
        local_ip = self.get_local_ip()
        
        print("\n" + "="*60)
        print("🎉 SERVER IS RUNNING!")
        print("="*60)
        print(f"\n📍 Local Access:")
        print(f"   http://localhost:{self.host_port}")
        print(f"   http://127.0.0.1:{self.host_port}")
        print(f"\n📍 Network Access (same network):")
        print(f"   http://{local_ip}:{self.host_port}")
        print(f"\n📚 API Documentation:")
        print(f"   http://{local_ip}:{self.host_port}/docs")
        print(f"   http://{local_ip}:{self.host_port}/redoc")
        
        print("\n📝 Useful Commands:")
        print(f"   View logs:     docker logs {self.container_name}")
        print(f"   Stop server:   docker stop {self.container_name}")
        print(f"   Start server:  docker start {self.container_name}")
        print(f"   Restart:       docker restart {self.container_name}")
        print(f"   Remove:        docker rm -f {self.container_name}")
        
        print("\n🌐 For EXTERNAL network access, see setup instructions below")
        print("="*60 + "\n")
    
    def setup(self):
        """Main setup process"""
        print("="*60)
        print("Raspberry Pi Docker Server Setup")
        print("="*60 + "\n")
        
        if not self.check_docker_installed():
            return False
        
        self.stop_existing_container()
        
        if not self.pull_image():
            return False
        
        if not self.start_container():
            return False
        
        if not self.check_container_health():
            return False
        
        self.display_info()
        return True


def main():
    manager = DockerServerManager()
    
    try:
        success = manager.setup()
        if success:
            print("✓ Setup complete! Your API server is ready.")
            sys.exit(0)
        else:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
