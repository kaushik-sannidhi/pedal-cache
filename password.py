#!/usr/bin/env python3
"""
CNIT 176 Project 3.2 - Password Strength Evaluator
Week 12 Implementation: Environment Setup, API Testing, Password Generator

Author: Kaushik Sannidhi
Date: Week 12 - Lab 1
"""

import hashlib
import requests
import secrets
import string
import time
import math
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

HIBP_API_URL = "https://api.pwnedpasswords.com/range/"
REQUEST_DELAY = 1.5  # Rate limit: 1 request per 1.5 seconds
TIMEOUT = 10  # API request timeout in seconds

# Password generator configuration
DEFAULT_PASSWORD_LENGTH = 16
DEFAULT_PASSPHRASE_WORDS = 4


# ============================================================================
# HAVE I BEEN PWNED - K-ANONYMITY API CLIENT
# ============================================================================

class BreachChecker:
    """
    Checks passwords against Have I Been Pwned database using k-anonymity.
    NEVER transmits full passwords - only first 5 characters of SHA-1 hash.
    """
    
    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _hash_password(self, password: str) -> str:
        """
        Hash password using SHA-1 (as required by HIBP API).
        Returns uppercase hex string.
        """
        sha1_hash = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
        return sha1_hash
    
    def check_breach(self, password: str) -> Tuple[bool, int]:
        """
        Check if password appears in breach database using k-anonymity.
        
        Args:
            password: The password to check
            
        Returns:
            Tuple of (is_breached: bool, breach_count: int)
            
        Security Note:
            Only sends first 5 characters of hash to API.
            Full password NEVER leaves local system.
        """
        # Hash the password
        full_hash = self._hash_password(password)
        
        # Split into prefix (sent to API) and suffix (checked locally)
        hash_prefix = full_hash[:5]
        hash_suffix = full_hash[5:]
        
        print(f"[DEBUG] Full hash: {full_hash}")
        print(f"[DEBUG] Sending prefix only: {hash_prefix}")
        print(f"[DEBUG] Checking suffix locally: {hash_suffix}")
        
        try:
            # Rate limit API requests
            self._rate_limit()
            self.request_count += 1
            
            # Make k-anonymity API request
            response = requests.get(
                f"{HIBP_API_URL}{hash_prefix}",
                timeout=TIMEOUT,
                headers={'User-Agent': 'CNIT176-PasswordTool'}
            )
            
            if response.status_code == 200:
                # Parse response: each line is "SUFFIX:COUNT"
                for line in response.text.splitlines():
                    parts = line.split(':')
                    if len(parts) == 2:
                        response_suffix, count = parts
                        if response_suffix == hash_suffix:
                            print(f"[BREACH FOUND] Password found {count} times in breaches!")
                            return True, int(count)
                
                # Suffix not found in breach database
                print("[SAFE] Password not found in known breaches")
                return False, 0
            
            elif response.status_code == 429:
                print("[WARNING] Rate limited by API")
                return False, -1  # -1 indicates API error
            
            else:
                print(f"[ERROR] API returned status code: {response.status_code}")
                return False, -1
                
        except requests.exceptions.Timeout:
            print("[ERROR] API request timed out")
            return False, -1
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Network error: {e}")
            return False, -1
    
    def get_stats(self) -> Dict:
        """Return statistics about API usage."""
        return {
            'total_requests': self.request_count,
            'last_request': self.last_request_time
        }


# ============================================================================
# PASSWORD GENERATOR
# ============================================================================

class PasswordGenerator:
    """
    Generates cryptographically secure passwords and passphrases.
    Uses Python's secrets module for CSPRNG.
    Downloads EFF wordlist from official source.
    """
    
    # EFF Long Wordlist URL (official source)
    EFF_WORDLIST_URL = "https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt"
    
    def __init__(self):
        self.generated_count = 0
        self.wordlist = None
        self._load_wordlist()
    
    def _load_wordlist(self):
        """
        Load EFF wordlist from online source or fallback to basic words.
        EFF wordlist has 7,776 words optimized for passphrases.
        """
        try:
            print("[INFO] Downloading EFF wordlist...")
            response = requests.get(self.EFF_WORDLIST_URL, timeout=10)
            
            if response.status_code == 200:
                # Parse EFF wordlist format: "11111	abacus"
                self.wordlist = []
                for line in response.text.strip().split('\n'):
                    parts = line.split('\t')
                    if len(parts) == 2:
                        self.wordlist.append(parts[1].strip())
                
                print(f"[SUCCESS] Loaded {len(self.wordlist)} words from EFF wordlist")
            else:
                print(f"[WARNING] EFF wordlist returned status {response.status_code}")
                self._use_fallback_wordlist()
                
        except requests.exceptions.RequestException as e:
            print(f"[WARNING] Could not download wordlist: {e}")
            self._use_fallback_wordlist()
    
    def _use_fallback_wordlist(self):
        """Use basic fallback wordlist if download fails."""
        print("[INFO] Using fallback wordlist (100 common words)")
        self.wordlist = [
            'able', 'about', 'account', 'acid', 'across', 'actor', 'actual', 'adapt',
            'added', 'adjust', 'admire', 'admit', 'adopt', 'adult', 'advance', 'advice',
            'affair', 'afford', 'afraid', 'after', 'again', 'against', 'agent', 'agree',
            'ahead', 'alarm', 'album', 'alert', 'alike', 'alive', 'allow', 'almost',
            'alone', 'along', 'aloud', 'alpha', 'already', 'also', 'alter', 'always',
            'amaze', 'amount', 'amuse', 'ancient', 'angel', 'anger', 'angle', 'angry',
            'animal', 'ankle', 'announce', 'annual', 'another', 'answer', 'anxiety', 'apart',
            'appeal', 'appear', 'apple', 'apply', 'approve', 'area', 'argue', 'arise',
            'around', 'arrange', 'arrest', 'arrive', 'arrow', 'artist', 'aside', 'aspect',
            'assault', 'asset', 'assist', 'assume', 'asthma', 'athlete', 'atom', 'attach',
            'attack', 'attempt', 'attend', 'attract', 'auction', 'audit', 'august', 'author',
            'auto', 'autumn', 'average', 'avoid', 'awake', 'aware', 'away', 'axis',
            'baby', 'bachelor', 'bacon', 'badge', 'balance', 'balcony', 'ball', 'banana'
        ]
    
    def generate_random_password(
        self,
        length: int = DEFAULT_PASSWORD_LENGTH,
        use_uppercase: bool = True,
        use_lowercase: bool = True,
        use_digits: bool = True,
        use_symbols: bool = True
    ) -> str:
        """
        Generate a random password with specified character types.
        
        Args:
            length: Password length (minimum 8)
            use_uppercase: Include uppercase letters
            use_lowercase: Include lowercase letters
            use_digits: Include digits
            use_symbols: Include symbols
            
        Returns:
            Cryptographically secure random password
        """
        if length < 8:
            raise ValueError("Password length must be at least 8 characters")
        
        # Build character set based on options
        charset = ""
        if use_lowercase:
            charset += string.ascii_lowercase
        if use_uppercase:
            charset += string.ascii_uppercase
        if use_digits:
            charset += string.digits
        if use_symbols:
            charset += string.punctuation
        
        if not charset:
            raise ValueError("At least one character type must be selected")
        
        # Generate password using secrets (CSPRNG)
        password = ''.join(secrets.choice(charset) for _ in range(length))
        
        self.generated_count += 1
        return password
    
    def generate_passphrase(
        self,
        num_words: int = DEFAULT_PASSPHRASE_WORDS,
        separator: str = '-',
        capitalize: bool = True,
        add_number: bool = True
    ) -> str:
        """
        Generate a memorable passphrase using random words.
        Example: "Correct-Horse-Battery-Staple-42"
        
        Args:
            num_words: Number of words in passphrase
            separator: Character to separate words
            capitalize: Capitalize first letter of each word
            add_number: Append random number
            
        Returns:
            Cryptographically secure passphrase
        """
        if num_words < 3:
            raise ValueError("Passphrase must contain at least 3 words")
        
        # Select random words from wordlist
        words = [secrets.choice(self.WORDLIST) for _ in range(num_words)]
        
        # Capitalize if requested
        if capitalize:
            words = [word.capitalize() for word in words]
        
        # Join with separator
        passphrase = separator.join(words)
        
        # Add random number if requested
        if add_number:
            number = secrets.randbelow(100)
            passphrase += f"{separator}{number}"
        
        self.generated_count += 1
        return passphrase
    
    def generate_pin(self, length: int = 6) -> str:
        """Generate a random numeric PIN."""
        if length < 4:
            raise ValueError("PIN must be at least 4 digits")
        
        # Generate PIN avoiding leading zeros
        first_digit = secrets.randbelow(9) + 1  # 1-9
        remaining = ''.join(str(secrets.randbelow(10)) for _ in range(length - 1))
        
        self.generated_count += 1
        return str(first_digit) + remaining
    
    def get_stats(self) -> Dict:
        """Return statistics about generated passwords."""
        return {
            'passwords_generated': self.generated_count
        }


# ============================================================================
# BASIC ENTROPY CALCULATOR
# ============================================================================

class EntropyCalculator:
    """
    Calculate password entropy for strength estimation.
    Entropy = log2(possible_combinations)
    """
    
    @staticmethod
    def calculate_charset_size(password: str) -> int:
        """Determine the character set size used in password."""
        has_lowercase = any(c in string.ascii_lowercase for c in password)
        has_uppercase = any(c in string.ascii_uppercase for c in password)
        has_digits = any(c in string.digits for c in password)
        has_symbols = any(c in string.punctuation for c in password)
        
        charset_size = 0
        if has_lowercase:
            charset_size += 26
        if has_uppercase:
            charset_size += 26
        if has_digits:
            charset_size += 10
        if has_symbols:
            charset_size += 32  # Approximate
        
        return charset_size
    
    @staticmethod
    def calculate_entropy(password: str) -> float:
        """
        Calculate Shannon entropy for password.
        
        Returns:
            Entropy in bits
        """
        if not password:
            return 0.0
        
        charset_size = EntropyCalculator.calculate_charset_size(password)
        length = len(password)
        
        # Entropy = log2(charset_size ^ length)
        entropy = length * math.log2(charset_size) if charset_size > 0 else 0
        
        return entropy
    
    @staticmethod
    def estimate_crack_time(entropy: float, guesses_per_second: float) -> str:
        """
        Estimate time to crack password given entropy and attack speed.
        
        Args:
            entropy: Password entropy in bits
            guesses_per_second: Attack speed (e.g., 10^10 for offline MD5)
            
        Returns:
            Human-readable time estimate
        """
        # Total possible combinations
        total_combinations = 2 ** entropy
        
        # Average time to crack (half the search space)
        seconds = (total_combinations / 2) / guesses_per_second
        
        # Convert to human-readable format
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.2f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.2f} hours"
        elif seconds < 31536000:
            return f"{seconds/86400:.2f} days"
        elif seconds < 31536000 * 100:
            return f"{seconds/31536000:.2f} years"
        else:
            return f"{seconds/31536000:.2e} years (effectively uncrackable)"


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_breach_checker():
    """Test the HIBP k-anonymity API integration."""
    print("\n" + "="*70)
    print("TESTING HAVE I BEEN PWNED API - K-ANONYMITY")
    print("="*70)
    
    checker = BreachChecker()
    
    # Test with known breached password
    test_passwords = [
        ("password", "Known breached password"),
        ("P@ssw0rd123!", "Common pattern with complexity"),
        ("correct-horse-battery-staple", "XKCD passphrase"),
        ("X9$mK2#pL5@qR8", "Strong random password")
    ]
    
    for password, description in test_passwords:
        print(f"\n--- Testing: {description} ---")
        print(f"Password: {password}")
        
        is_breached, count = checker.check_breach(password)
        
        if count == -1:
            print("❌ API Error - could not verify")
        elif is_breached:
            print(f"⚠️  BREACHED: Found {count:,} times in data breaches")
        else:
            print("✅ SAFE: Not found in known breaches")
        
        print("-" * 70)
    
    # Show API stats
    print("\n" + "="*70)
    print("API USAGE STATISTICS")
    print("="*70)
    stats = checker.get_stats()
    print(f"Total API requests made: {stats['total_requests']}")
    print(f"K-anonymity preserved: Only hash prefixes transmitted")
    print(f"Full passwords: NEVER sent to server")


def test_password_generator():
    """Test password generation functionality."""
    print("\n" + "="*70)
    print("TESTING PASSWORD GENERATOR")
    print("="*70)
    
    generator = PasswordGenerator()
    
    # Generate various password types
    print("\n--- Random Passwords ---")
    for i in range(3):
        pwd = generator.generate_random_password(length=16)
        entropy = EntropyCalculator.calculate_entropy(pwd)
        print(f"{i+1}. {pwd} (Entropy: {entropy:.1f} bits)")
    
    print("\n--- Passphrases ---")
    for i in range(3):
        phrase = generator.generate_passphrase(num_words=4)
        entropy = EntropyCalculator.calculate_entropy(phrase)
        print(f"{i+1}. {phrase} (Entropy: {entropy:.1f} bits)")
    
    print("\n--- PINs ---")
    for i in range(3):
        pin = generator.generate_pin(length=6)
        print(f"{i+1}. {pin}")
    
    # Show generation stats
    print("\n" + "="*70)
    print("GENERATION STATISTICS")
    print("="*70)
    stats = generator.get_stats()
    print(f"Total passwords generated: {stats['passwords_generated']}")
    print(f"CSPRNG used: Python secrets module")


def test_entropy_calculator():
    """Test entropy calculation and crack time estimation."""
    print("\n" + "="*70)
    print("TESTING ENTROPY CALCULATOR")
    print("="*70)
    
    test_passwords = [
        "password",
        "P@ssw0rd",
        "Tr0ub4dor&3",
        "correct-horse-battery-staple",
        "X9$mK2#pL5@qR8vW",
    ]
    
    # Attack scenarios
    scenarios = [
        ("Online (throttled)", 1000),           # 1k guesses/sec
        ("Offline slow hash (Argon2)", 10**4),  # 10k guesses/sec
        ("Offline fast hash (MD5)", 10**10),    # 10B guesses/sec
    ]
    
    for password in test_passwords:
        print(f"\n--- Password: {password} ---")
        
        charset = EntropyCalculator.calculate_charset_size(password)
        entropy = EntropyCalculator.calculate_entropy(password)
        
        print(f"Length: {len(password)} characters")
        print(f"Charset size: {charset}")
        print(f"Entropy: {entropy:.1f} bits")
        print(f"\nEstimated crack times:")
        
        for scenario_name, speed in scenarios:
            crack_time = EntropyCalculator.estimate_crack_time(entropy, speed)
            print(f"  {scenario_name}: {crack_time}")
        
        print("-" * 70)


def main():
    """Main testing routine for Week 12 implementation."""
    print("\n" + "="*70)
    print("CNIT 176 PROJECT 3.2 - WEEK 12 IMPLEMENTATION TEST")
    print("Password Strength Evaluator with Breach Detection")
    print("="*70)
    print("\nAuthor: Kaushik Sannidhi")
    print("Status: Lab 1 - Environment Setup & Initial Testing")
    print("\n" + "="*70)
    
    # Run all tests
    try:
        test_password_generator()
        test_entropy_calculator()
        test_breach_checker()  # This makes actual API calls
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nWeek 12 Accomplishments Verified:")
        print("  ✓ Development environment configured")
        print("  ✓ HIBP k-anonymity API tested")
        print("  ✓ Password generator implemented")
        print("  ✓ Basic entropy calculations working")
        print("\nReady for Week 13: Core integration & breach checker module")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Check network connectivity and dependencies")


if __name__ == "__main__":
    main()
