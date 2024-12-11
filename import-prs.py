import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Generator, Any
import os
from pathlib import Path
import random  # Added for jitter calculation

class GitHubPRExtractor:
    def __init__(self, owner: str, repo: str, token: str, output_dir: str = "output"):
        """
        Initialize the GitHub PR data extractor using GraphQL API.
        
        Args:
            owner: GitHub repository owner
            repo: Repository name
            token: GitHub personal access token
            output_dir: Directory to store output files
        """
        self.owner = owner
        self.repo = repo
        self.token = token
        self.graphql_url = "https://api.github.com/graphql"
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists before setting up logging
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-PR-Extractor"
        })
        
        # Set up logging after directory creation
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging with both file and console handlers."""
        self.logger = logging.getLogger("github_pr_extractor")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate log messages
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler(self.output_dir / "extraction.log")
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _handle_rate_limit(self, response_headers: Dict) -> None:
        """Handle GitHub API rate limiting for GraphQL."""
        remaining = int(response_headers.get('X-RateLimit-Remaining', 0))
        if remaining <= 1:
            reset_time = int(response_headers.get('X-RateLimit-Reset', 0))
            if reset_time:
                wait_time = reset_time - int(time.time()) + 1
                if wait_time > 0:
                    self.logger.warning(f"Rate limit near zero. Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)

    def _make_graphql_request(self, query: str, variables: Dict[str, Any]) -> Dict:
        """Make a GraphQL request to GitHub API with exponential backoff retry logic."""
        max_retries = 5  # Increased max retries
        base_delay = 5
        max_delay = 120  # Maximum delay of 2 minutes
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self.graphql_url,
                    json={'query': query, 'variables': variables}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'errors' in result:
                        self.logger.error(f"GraphQL errors: {result['errors']}")
                        raise Exception("GraphQL query failed")
                    return result['data']
                elif response.status_code == 403:
                    self._handle_rate_limit(response.headers)
                    continue
                
                # Calculate exponential backoff delay
                delay = min(max_delay, base_delay * (2 ** attempt))
                jitter = random.uniform(0, 0.1 * delay)  # Add 0-10% jitter
                total_delay = delay + jitter
                
                self.logger.warning(
                    f"Request failed with status {response.status_code}. "
                    f"Attempt {attempt + 1}/{max_retries}. "
                    f"Retrying in {total_delay:.1f} seconds..."
                )
                
                time.sleep(total_delay)
                    
            except requests.RequestException as e:
                self.logger.error(
                    f"Request error: {str(e)}. "
                    f"Attempt {attempt + 1}/{max_retries}"
                )
                if attempt < max_retries - 1:
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0, 0.1 * delay)
                    total_delay = delay + jitter
                    self.logger.info(f"Retrying in {total_delay:.1f} seconds...")
                    time.sleep(total_delay)
                else:
                    raise
        
        raise Exception(f"Failed to make request after {max_retries} attempts")

    def _get_pull_requests_batch(self, cursor: Optional[str] = None) -> Dict:
        """Get a batch of pull requests using GraphQL."""
        query = """
        query($owner: String!, $repo: String!, $cursor: String) {
            repository(owner: $owner, name: $repo) {
                pullRequests(
                    first: 100,
                    after: $cursor,
                    orderBy: {field: CREATED_AT, direction: ASC},
                    states: [CLOSED, MERGED]
                ) {
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                    nodes {
                        number
                        createdAt
                        closedAt
                        mergedAt
                        state
                        author {
                            login
                        }
                        baseRef {
                            name
                        }
                        additions
                        deletions
                        changedFiles
                    }
                }
            }
        }
        """
        
        variables = {
            "owner": self.owner,
            "repo": self.repo,
            "cursor": cursor
        }
        
        return self._make_graphql_request(query, variables)

    def extract_all_prs(self) -> None:
        """Extract all closed pull request data using GraphQL API and save to JSON file."""
        self.logger.info(f"Starting PR extraction for {self.owner}/{self.repo}")
        
        try:
            all_prs = []
            cursor = None
            batch_count = 0
            
            while True:
                batch_count += 1
                self.logger.info(f"Fetching batch {batch_count}...")
                
                data = self._get_pull_requests_batch(cursor)
                pull_requests = data['repository']['pullRequests']
                
                # Add PRs from this batch
                all_prs.extend(pull_requests['nodes'])
                self.logger.info(f"Processed {len(all_prs)} PRs so far")
                
                # Save intermediate results every 5 batches
                if batch_count % 5 == 0:
                    self._save_intermediate_results(all_prs)
                
                # Check if there are more pages
                page_info = pull_requests['pageInfo']
                if not page_info['hasNextPage']:
                    break
                
                cursor = page_info['endCursor']
            
            # Save final results
            self._save_final_results(all_prs)
            
        except Exception as e:
            self.logger.error(f"Error during PR extraction: {str(e)}")
            raise

    def _save_intermediate_results(self, prs: List[Dict]) -> None:
        """Save intermediate results to prevent data loss."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"prs_intermediate_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(prs, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved intermediate results to {filename}")

    def _save_final_results(self, prs: List[Dict]) -> None:
        """Save final results to JSON file."""
        filename = self.output_dir / "pull_requests.json"
        
        # Calculate and add duration for each PR
        for pr in prs:
            if pr['createdAt'] and pr['closedAt']:
                created = datetime.fromisoformat(pr['createdAt'].replace('Z', '+00:00'))
                closed = datetime.fromisoformat(pr['closedAt'].replace('Z', '+00:00'))
                pr['duration_hours'] = (closed - created).total_seconds() / 3600
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(prs, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Successfully saved {len(prs)} PRs to {filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract GitHub PR data')
    parser.add_argument('--owner', required=True, help='Repository owner')
    parser.add_argument('--repo', required=True, help='Repository name')
    parser.add_argument('--token', required=True, help='GitHub personal access token')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    extractor = GitHubPRExtractor(
        owner=args.owner,
        repo=args.repo,
        token=args.token,
        output_dir=args.output_dir
    )
    
    extractor.extract_all_prs()