# Geo ETL

## Main Files 

Airflow Dags
- `dags/dags/geotl_dag.py` 
  DAG FILE which automates the Data processing of files
- `dags/tests/dags/test_dag_integrity.py`
  Just basic test [Can ignore this if required]

Spatial Analysis Codes:
- `geotl/v0_neighbour_slow.py` Naive method of solving this problem or v0 version

- `geotl/main.py` Current solution to problem using BallTree instead of kd_tree

- `geotl/main.ipynb`
    Has some visualizations in Kepler.gl and other exploration code

