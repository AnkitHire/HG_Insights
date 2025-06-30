import time
import schedule
import subprocess
import configparser

config = configparser.ConfigParser()
config.read('C:/Users/ankit.h/Pictures/Saved Pictures/personal/HG/config/config.ini')

def run_pipeline():
    print("Starting pipeline execution...")
    
    
    # Load to staging
    print("Loading to staging...")
    subprocess.run(["python", "01_load_to_staging.py"], cwd="C:/Users/ankit.h/Pictures/Saved Pictures/personal/HG/scripts")
    
    # Transform data
    print("Transforming data...")
    subprocess.run(["python", "02_transform_data.py"], cwd="C:/Users/ankit.h/Pictures/Saved Pictures/personal/HG/scripts")
    
    print("Pipeline execution completed!")

def main():
    # Run immediately
    run_pipeline()
    
    # Schedule hourly runs
    schedule.every(int(config['pipeline']['schedule_minutes'])).minutes.do(run_pipeline)
    
    print(f"Pipeline scheduled to run every {config['pipeline']['schedule_minutes']} minutes...")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()