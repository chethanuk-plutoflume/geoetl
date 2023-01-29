import datetime

from airflow import models
from airflow.hooks.base import BaseHook
from airflow.providers.google.cloud.operators import dataproc
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.dates import days_ago
from airflow.utils.state import State
from airflow.utils.task_group import TaskGroup
from astro import sql as aql
from astro.files import File
from astro.table import Table

# PROJECT_NAME = "{{var.value.gcp_project}}"
PROJECT_NAME = "TEST_PROJ"

# Dataproc configs
# BUCKET_NAME = "{{var.value.gcs_bucket}}"
BUCKET_NAME = "data-lake"
PYSPARK_JAR = "gs://spark-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.26.0.jar"
PROCESSING_PYTHON_FILE = f"gs://{BUCKET_NAME}/geoetl.py"
REGION = "europe-west1"
ZONE = "europe-west1-b"

# Cluster definition

CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "n1-standard-4",
        "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 1024},
    },
    "worker_config": {
        "num_instances": 2,
        "machine_type_uri": "n1-standard-4",
        "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 1024},
    },
}

TIMEOUT = {"seconds": 1 * 24 * 60 * 60}


# Slack error notification example taken from Kaxil Naik's blog on Slack Integration:
# https://medium.com/datareply/integrating-slack-alerts-in-airflow-c9dcd155105
def on_failure_callback(context):
    ti = context.get("task_instance")
    slack_msg = f"""
            :red_circle: Task Failed.
            *Task*: {ti.task_id}
            *Dag*: {ti.dag_id}
            *Execution Time*: {context.get('execution_date')}
            *Log Url*: {ti.log_url}
            """
    slack_webhook_token = BaseHook.get_connection("slack_connection").password
    slack_error = SlackWebhookOperator(
        task_id="post_slack_error",
        http_conn_id="slack_connection",
        channel="#airflow-alerts",
        webhook_token=slack_webhook_token,
        message=slack_msg,
    )
    slack_error.execute(context)


default_args = {"project_id": PROJECT_NAME, "region": REGION, "BUCKET_NAME": BUCKET_NAME}

with models.DAG(
    "geotl",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
) as dag:

    ds = "{{ ds_nodash }}"
    DATAPROC_CLUSTER_NAME = f"geoetl-process-{ds}"

    # Jobs definitions
    # [START how_to_cloud_dataproc_pyspark_config]
    PYSPARK_JOB = {
        "reference": {"project_id": PROJECT_NAME},
        "placement": {"cluster_name": CLUSTER_NAME},
        "pyspark_job": {"main_python_file_uri": PROCESSING_PYTHON_FILE},
    }
    # [END how_to_cloud_dataproc_pyspark_config]

    locations_path = f"gs://{BUCKET_NAME}/data/dt={ds}/ms_hinds_locations.csv"
    buildings_path = f"gs://{BUCKET_NAME}/data/dt={ds}/ms_hinds_buildings.geojson.zip"
    parcels_path = f"gs://{BUCKET_NAME}/data/dt={ds}/ms_hinds_parcels.ndgeojson.gz"

    # START SENSORS
    validate_locations_exists = GCSObjectExistenceSensor(
        task_id="validate_locations_exists",
        bucket=BUCKET_NAME,
        object=locations_path.split("gs://{BUCKET_NAME}/")[1],
    )
    validate_buildings_exists = GCSObjectExistenceSensor(
        task_id="validate_buildings_exists",
        bucket=BUCKET_NAME,
        object=buildings_path.split("gs://{BUCKET_NAME}/")[1],
    )
    validate_parcels_exists = GCSObjectExistenceSensor(
        task_id="validate_parcels_exists",
        bucket=BUCKET_NAME,
        object=parcels_path.split("gs://{BUCKET_NAME}/")[1],
    )
    # END SENSORS

    df_locations = aql.load_file(
        task_id="load_locations",
        input_file=File(path=locations_path),
        columns_names_capitalization="upper",
        output_table=Table(conn_id="sqlite_default"),
    )

    df_buildings = aql.load_file(
        task_id="load_locations",
        input_file=File(path=buildings_path),
        columns_names_capitalization="upper",
        output_table=Table(conn_id="sqlite_default"),
    )

    @aql.transform()
    def check_locations_count(input_table: Table):
        return """
             WITH tmp AS (SELECT LEN(f_ziploc) as len FROM {{input_table}})
             SELECT len, COUNT(1) as num_records
             FROM tmp
             GROUP BY len
        """

    check_locations_count(
        input_table=df_locations,
        output_table=Table(name="locations_zip_count"),
    )
    # TODO: Check data quality
    # TODO: Fail if count is not greater than certain limit
    # TODO:
    aql.cleanup()

    # [START CREATE DataProc Cluster]
    create_cluster = DataprocCreateClusterOperator(
        task_id="create_cluster",
        cluster_config=CLUSTER_CONFIG,
        region=REGION,
        cluster_name=CLUSTER_NAME,
    )
    # [END CREATE]

    # [START submit_job_to_cluster_operator]
    pyspark_task = DataprocSubmitJobOperator(task_id="pyspark_task", job=PYSPARK_JOB)
    # [END submit_job_to_cluster_operator]

    # [START delete_cluster_operator]
    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_cluster",
        cluster_name=CLUSTER_NAME,
        region=REGION,
        trigger_rule=TriggerRule.ALL_DONE,
    )
    # [END delete_cluster_operator]

    output_loc = "todo"
    validate_output_exists = GCSObjectExistenceSensor(
        task_id="validate_locations_exists",
        bucket=BUCKET_NAME,
        object=output_loc.split("gs://{BUCKET_NAME}/")[1],
    )

    # SENSORS >> PreDataQuality >> CreateCluster > SubmitJob >> DeleteCluster >> OutputSensor >> PostDQ >> ALERT

    (
        [validate_locations_exists, validate_buildings_exists, validate_parcels_exists]
        >> [df_locations, df_buildings, df_parcels]
        >> [check_locations_count, check_buildings_count, check_parcel_count]
        >> create_cluster
        >> pyspark_task
        >> delete_cluster
        >> validate_output_exists
        >> df_output
        >> check_output
        >> send_alerts
    )
