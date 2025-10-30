#!/usr/bin/env python3
"""
SQL Dataset Migration Script

This script migrates CSV datasets to SQL format.
Usage:
    python sql-dataset-migrate.py [--update]
    
Options:
    --update    Force update existing SQL files
"""

import sys
import argparse
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.csv_to_sql_migrator import CSVToSQLMigrator


def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description='Migrate CSV datasets to SQL format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sql-dataset-migrate.py          # Migrate all CSV files (skip existing SQL)
  python sql-dataset-migrate.py --update # Update all SQL files from CSV
        """
    )
    
    parser.add_argument(
        '--update',
        action='store_true',
        help='Force update existing SQL files'
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset',
        help='Directory containing CSV files (default: dataset)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CSV to SQL Dataset Migrator")
    print("=" * 60)
    print()
    
    try:
        # Create migrator instance
        migrator = CSVToSQLMigrator(dataset_dir=args.dataset_dir)
        
        # Perform migration
        sql_files = migrator.migrate_all(force_update=args.update)
        
        if sql_files:
            print("\n" + "=" * 60)
            print("Success! SQL files generated:")
            print("=" * 60)
            for sql_file in sql_files:
                print(f"  ✓ {sql_file.name}")
        else:
            print("\n" + "=" * 60)
            if args.update:
                print("No CSV files found to process")
            else:
                print("All SQL files are up to date")
                print("Use --update flag to force regeneration")
            print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

