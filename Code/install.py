import os
import os.path
import ssl
import stat
import subprocess
import sys
import platform
import shutil
import datetime
import time
import argparse
import logging
from pathlib import Path
import ctypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('cert_manager')

# Define permission constants for different platforms
if platform.system() == 'Windows':
    # Windows permissions (everyone can read and execute)
    CERT_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH
else:
    # Unix-like permissions (rwxr-xr-x)
    CERT_PERMISSIONS = (
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |  # Owner: rwx
        stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP |  # Group: r-x
        stat.S_IROTH | stat.S_IXOTH                   # Others: r-x
    )

class CertificateManager:
    """Advanced certificate management for OCR applications."""
    
    def __init__(self, args):
        """Initialize the certificate manager with command line arguments."""
        self.args = args
        self.backup_dir = None
        self.is_admin = self._check_admin_privileges()
        self.platform = platform.system()
        
        # Get SSL paths
        ssl_paths = ssl.get_default_verify_paths()
        self.openssl_dir, self.openssl_cafile = os.path.split(ssl_paths.openssl_cafile)
        self.cafile_path = ssl_paths.openssl_cafile
        
        # Create backup directory if needed
        if args.backup:
            self._setup_backup_directory()
    
    def _check_admin_privileges(self):
        """Check if the script is running with administrative privileges."""
        try:
            if platform.system() == 'Windows':
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                # For Unix-like systems, check if UID is 0 (root)
                return os.geteuid() == 0
        except:
            return False
    
    def _setup_backup_directory(self):
        """Create a backup directory for certificate files."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"cert_backup_{timestamp}"
        )
        os.makedirs(self.backup_dir, exist_ok=True)
        logger.info(f"Created backup directory: {self.backup_dir}")
    
    def _backup_existing_cert(self):
        """Backup the existing certificate file if it exists."""
        if not self.args.backup or not os.path.exists(self.cafile_path):
            return
            
        backup_file = os.path.join(self.backup_dir, os.path.basename(self.cafile_path))
        shutil.copy2(self.cafile_path, backup_file)
        logger.info(f"Backed up existing certificate to {backup_file}")
    
    def _show_progress(self, message, delay=0.05, steps=20):
        """Display a progress bar with the given message."""
        if self.args.quiet:
            return
            
        sys.stdout.write(f"{message} ")
        for i in range(steps):
            sys.stdout.write("█")
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write(" Done!\n")
    
    def update_certifi(self):
        """Update the certifi package to the latest version."""
        logger.info("Updating certifi package...")
        self._show_progress("Installing latest certificate bundle")
        
        try:
            subprocess.check_call([
                sys.executable, "-E", "-s", "-m", "pip", 
                "install", "--upgrade", "certifi"
            ], stdout=subprocess.DEVNULL if self.args.quiet else None)
            logger.info("Successfully updated certifi package")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update certifi: {e}")
            return False
    
    def install_certificates(self):
        """Install the certificates from certifi to the system."""
        # Import certifi after updating it
        import certifi
        
        logger.info(f"Installing certificates to {self.cafile_path}")
        self._show_progress("Configuring system certificates")
        
        # Backup existing certificate
        self._backup_existing_cert()
        
        # Change to the SSL directory
        original_dir = os.getcwd()
        os.chdir(self.openssl_dir)
        
        try:
            # Get the path to certifi's certificates
            certifi_path = certifi.where()
            rel_certifi_path = os.path.relpath(certifi_path)
            
            # Remove existing certificate file if it exists
            if os.path.exists(self.openssl_cafile):
                os.remove(self.openssl_cafile)
            
            # Create link or copy based on platform and permissions
            if self.platform == 'Windows' and not self.is_admin:
                # On Windows without admin rights, copy the file instead of symlinking
                logger.info("Windows detected without admin rights, copying certificates")
                shutil.copy2(certifi_path, self.openssl_cafile)
            else:
                # Create symbolic link on Unix or Windows with admin rights
                logger.info("Creating symbolic link to certifi certificates")
                os.symlink(rel_certifi_path, self.openssl_cafile)
            
            # Set appropriate permissions
            os.chmod(self.openssl_cafile, CERT_PERMISSIONS)
            logger.info("Certificate permissions set successfully")
            
            return True
        except Exception as e:
            logger.error(f"Error installing certificates: {e}")
            return False
        finally:
            # Return to the original directory
            os.chdir(original_dir)
    
    def verify_installation(self):
        """Verify that the certificates were installed correctly."""
        logger.info("Verifying certificate installation...")
        self._show_progress("Validating certificate configuration")
        
        try:
            # Try to make a secure connection to verify certificates
            import urllib.request
            urllib.request.urlopen("https://www.python.org/")
            logger.info("Certificate verification successful!")
            return True
        except Exception as e:
            logger.error(f"Certificate verification failed: {e}")
            return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced SSL Certificate Manager for OCR Applications",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--backup", "-b", action="store_true",
        help="Backup existing certificates before replacing"
    )
    
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output and run silently"
    )
    
    parser.add_argument(
        "--verify", "-v", action="store_true",
        help="Verify certificate installation after updating"
    )
    
    parser.add_argument(
        "--log-file", "-l", type=str,
        help="Save log output to the specified file"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the certificate manager."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure file logging if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Print header
    if not args.quiet:
        print("\n" + "=" * 70)
        print(" Enhanced SSL Certificate Manager for OCR Applications ".center(70, "="))
        print("=" * 70 + "\n")
    
    # Initialize and run the certificate manager
    cert_manager = CertificateManager(args)
    
    # Check for admin rights and warn if needed
    if platform.system() == 'Windows' and not cert_manager.is_admin:
        logger.warning("Running without administrator privileges on Windows.")
        logger.warning("Will use file copy instead of symbolic links.")
        if not args.quiet:
            print("\nNOTE: For best results on Windows, run as Administrator.\n")
    
    # Update certifi package
    if not cert_manager.update_certifi():
        logger.error("Failed to update certifi package. Aborting.")
        return 1
    
    # Install certificates
    if not cert_manager.install_certificates():
        logger.error("Failed to install certificates. Aborting.")
        return 1
    
    # Verify installation if requested
    if args.verify:
        if not cert_manager.verify_installation():
            logger.warning("Certificate verification failed, but installation completed.")
    
    # Print success message
    if not args.quiet:
        print("\n" + "-" * 70)
        print(" Certificate installation completed successfully! ".center(70, "-"))
        print("-" * 70 + "\n")
        
        if cert_manager.backup_dir:
            print(f"Backup saved to: {cert_manager.backup_dir}")
        
        print("\nYour OCR applications can now make secure connections.")
        print("=" * 70 + "\n")
    
    logger.info("Certificate installation completed successfully")
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)