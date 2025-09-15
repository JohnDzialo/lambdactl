#!/usr/bin/env python3
"""
Lambda Labs Cluster Management CLI

A command-line interface for deploying clusters, running load tests,
and collecting metrics from Lambda Labs GPU instances.

Usage:
    python cli.py --help
    python cli.py deploy --help
    python cli.py test --help
    python cli.py metrics --help
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import click

from lambda_api import LambdaLabsAPI, Instance, InstanceRequest
from remote_executor import RemoteLoadTestExecutor, MonitoringConfig


# Global configuration
CONFIG_FILE = Path.home() / ".lambda-deploy" / "config.json"
LOGS_DIR = Path.home() / ".lambda-deploy" / "logs"
METRICS_DIR = Path.home() / ".lambda-deploy" / "metrics"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "lambda-deploy.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)


def load_config() -> Dict:
    """Load configuration from file"""
    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        return {}


def save_config(config: Dict):
    """Save configuration to file"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        click.echo(f"Error saving config: {e}", err=True)


def get_api_client() -> LambdaLabsAPI:
    """Get authenticated API client"""
    config = load_config()
    api_key = config.get("api_key")

    if not api_key:
        click.echo(
            "API key not configured. Use 'lambda-deploy configure' first.", err=True
        )
        sys.exit(1)

    return LambdaLabsAPI(api_key)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, verbose):
    """Lambda Labs Cluster Management CLI"""
    ctx.ensure_object(dict)
    ctx.obj["logger"] = setup_logging(verbose)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option("--api-key", prompt=True, hide_input=True, help="Lambda Labs API key")
def configure(api_key):
    """Configure Lambda Labs API credentials"""
    config = load_config()
    config["api_key"] = api_key
    save_config(config)
    click.echo("Configuration saved successfully!")


@cli.command()
@click.option("--available", is_flag=True, help="Show only available instance types")
def list_types(available):
    """List available instance types"""
    api = get_api_client()

    try:
        types = api.list_instance_types()

        click.echo("\nInstance Types:")
        click.echo("-" * 50)

        for instance_type in types.keys():
            name = types[instance_type].get("instance_type", {}).get("name", "Unknown")
            description = (
                types[instance_type]
                .get("instance_type", {})
                .get("description", "No description")
            )
            price = (
                types[instance_type]
                .get("instance_type", {})
                .get("price_cents_per_hour", 0)
                / 100
            )
            specs = types[instance_type].get("instance_type", {}).get("specs", {})
            regions_with_capacity = types[instance_type].get(
                "regions_with_capacity_available", []
            )

            if available:
                if not regions_with_capacity:
                    continue

            regions = ", ".join(
                [region.get("name", "Unknown") for region in regions_with_capacity]
            )
            spec = ""

            click.echo(f"Name: {name}")
            click.echo(f"Description: {description}")
            click.echo(f"Price: ${price:.2f}/hour")
            click.echo(f"Regions: {regions}")
            click.echo(f"GPUs: {specs.get("gpus", "")}")
            click.echo(f"Mem: {specs.get("memory_gib", "")} GB")
            click.echo(f"Storage: {specs.get("storage_gib", "")} GB")
            click.echo(f"vCPUs: {specs.get("vcpus", "")}")
            click.echo("-" * 30)

    except Exception as e:
        click.echo(f"Error listing instance types: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_instances():
    """List all instances"""
    api = get_api_client()

    try:
        instances = api.list_instances()

        if not instances:
            click.echo("No instances found.")
            return

        click.echo(f"\nFound {len(instances)} instances:")
        click.echo("-" * 80)

        for instance in instances:
            name = instance.get("name", "Unnamed")
            instance_id = instance.get("id", "Unknown")
            status = instance.get("status", "Unknown")
            instance_type = instance.get("instance_type", {}).get("name", "Unknown")
            ip = instance.get("ip", "No IP")

            click.echo(f"Name: {name}")
            click.echo(f"ID: {instance_id}")
            click.echo(f"Type: {instance_type}")
            click.echo(f"Status: {status}")
            click.echo(f"IP: {ip}")
            click.echo("-" * 50)

    except Exception as e:
        click.echo(f"Error listing instances: {e}", err=True)
        sys.exit(1)


@cli.group()
def deploy():
    """Cluster deployment commands"""
    pass


@deploy.command()
@click.option("--type", "-t", default="gpu_1x_a100", help="Instance type to deploy")
@click.option("--count", "-c", default=2, help="Number of instances to deploy")
@click.option("--name-prefix", default="cluster", help="Instance name prefix")
@click.option("--region", "-r", required=True, help="Region to deploy instances in")
@click.option("--ssh-key", required=True, help="SSH key name to use")
@click.option(
    "--file-systems", help="Comma-separated list of file system names (optional)"
)
@click.option("--hostname", help="Custom hostname for instances (optional)")
@click.option("--image", help="Custom image ID (optional)")
@click.option(
    "--user-data", help="User data script to run on instance startup (optional)"
)
@click.option(
    "--tags", help="Tags for instances, format: key=value,key=value (optional)"
)
@click.option(
    "--firewall-rules", help="Firewall ruleset IDs, comma-separated (optional)"
)
@click.option("--wait/--no-wait", default=True, help="Wait for instances to be ready")
@click.option(
    "--timeout", default=15, help="Timeout in minutes for instances to be ready"
)
@click.pass_context
def cluster(
    ctx,
    type,
    count,
    name_prefix,
    region,
    ssh_key,
    file_systems,
    hostname,
    image,
    user_data,
    tags,
    firewall_rules,
    wait,
    timeout,
):
    """Deploy a cluster of GPU instances"""
    logger = ctx.obj["logger"]
    api = get_api_client()

    logger.info(f"Deploying {count} instances of type {type}")

    # Store deployment info
    deployment_info = {
        "timestamp": time.time(),
        "instance_type": type,
        "instance_count": count,
        "name_prefix": name_prefix,
        "instances": [],
    }

    try:
        # Parse optional parameters
        file_system_names = file_systems.split(",") if file_systems else None

        # Parse image parameter (string ID to dict format)
        image_dict = None
        if image:
            image_dict = {"id": image}

        # Parse tags parameter (key=value,key=value format)
        tags_list = None
        if tags:
            try:
                tags_list = []
                for tag in tags.split(","):
                    key, value = tag.split("=", 1)
                    tags_list.append({key: value})
            except ValueError:
                click.echo(
                    f"Invalid tags format. Use key=value,key=value format.", err=True
                )
                sys.exit(1)

        # Parse firewall rules (list of IDs to dict format)
        firewall_rulesets = None
        if firewall_rules:
            firewall_rulesets = [
                {"id": rule_id.strip()} for rule_id in firewall_rules.split(",")
            ]

        # Deploy instances in parallel
        instances_created = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            for i in range(count):
                instance_name = f"{name_prefix}-{i+1}"

                # Create InstanceRequest object
                instance_request = InstanceRequest(
                    instance_type_name=type,
                    name=instance_name,
                    region_name=region,
                    ssh_key_names=[ssh_key],
                    file_system_names=file_system_names,
                    hostname=hostname,
                    image=image_dict,
                    user_data=user_data,
                    tags=tags_list,
                    firewall_rulesets=firewall_rulesets,
                )

                future = executor.submit(api.create_instance, instance_request)
                futures.append((future, instance_name))

            for future, instance_name in futures:
                try:
                    result = future.result()
                    if "data" in result:
                        instance = result["data"]
                        instances_created.append(instance)
                        deployment_info["instances"].append(instance)
                        logger.info(
                            f"Created instance: {instance_name} (ID: {instance.get('id', 'Unknown')})"
                        )
                    else:
                        logger.error(
                            f"Failed to create instance {instance_name}: {result}"
                        )
                except Exception as e:
                    logger.error(f"Error creating instance {instance_name}: {e}")

        if not instances_created:
            click.echo("Failed to create any instances", err=True)
            sys.exit(1)

        # Save deployment info
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        deployment_file = METRICS_DIR / f"deployment_{int(time.time())}.json"
        with open(deployment_file, "w") as f:
            json.dump(deployment_info, f, indent=2)

        click.echo(f"Successfully created {len(instances_created)} instances")
        click.echo(f"Deployment info saved to: {deployment_file}")

        if wait:
            click.echo(f"Waiting up to {timeout} minutes for instances to be ready...")

            ready_instances = []
            with ThreadPoolExecutor(max_workers=len(instances_created)) as executor:
                wait_futures = []

                for instance in instances_created:
                    instance_id = instance.get("id")
                    if instance_id:
                        future = executor.submit(
                            api.wait_for_instance_status,
                            instance_id,
                            "running",
                            timeout,
                        )
                        wait_futures.append((future, instance))

                for future, instance in wait_futures:
                    try:
                        if future.result():
                            ready_instances.append(instance)
                            logger.info(
                                f"Instance {instance.get('name', 'Unknown')} is ready"
                            )
                        else:
                            logger.warning(
                                f"Instance {instance.get('name', 'Unknown')} not ready in time"
                            )
                    except Exception as e:
                        logger.error(f"Error waiting for instance: {e}")

            click.echo(
                f"{len(ready_instances)}/{len(instances_created)} instances are ready"
            )

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        click.echo(f"Deployment failed: {e}", err=True)
        sys.exit(1)


@cli.group()
def test():
    """Load testing commands"""
    pass


@test.command()
@click.option("--instances", help="Comma-separated list of instance IDs to test")
@click.option(
    "--deployment-file",
    type=click.Path(exists=True),
    help="Use instances from deployment file",
)
@click.option("--ssh-key", required=True, help="SSH private key file path")
@click.option("--username", default="ubuntu", help="SSH username for instances")
@click.option("--quick", is_flag=True, help="Run quick GPU load tests")
@click.option(
    "--monitoring/--no-monitoring",
    default=True,
    help="Enable/disable resource monitoring (default: enabled)",
)
@click.option(
    "--monitoring-interval",
    default=60.0,
    help="Resource monitoring interval in seconds (default: 60.0)",
)
@click.option("--max-workers", default=5, help="Maximum parallel workers")
@click.option("--output", help="Output file for aggregated results")
@click.pass_context
def load(
    ctx,
    instances,
    deployment_file,
    ssh_key,
    username,
    quick,
    monitoring,
    monitoring_interval,
    max_workers,
    output,
):
    """Run GPU load tests on instances using remote execution"""
    logger = ctx.obj["logger"]
    api = get_api_client()

    # Get instance list and their IPs
    instance_data = []

    if deployment_file:
        try:
            with open(deployment_file, "r") as f:
                deployment = json.load(f)
                for inst in deployment.get("instances", []):
                    for instance_id in inst.get("instance_ids", []):
                        try:
                            instance_details = api.get_instance(instance_id)
                            print(instance_details)
                            if instance_details.get("ip"):
                                instance_data.append(
                                    {
                                        "id": instance_id,
                                        "name": instance_details.get("name", "Unknown"),
                                        "ip": instance_details["ip"],
                                    }
                                )
                        except Exception as e:
                            logger.warning(
                                f"Could not get details for instance {instance_id}: {e}"
                            )
        except Exception as e:
            click.echo(f"Error reading deployment file: {e}", err=True)
            sys.exit(1)
    elif instances:
        instance_ids = [id.strip() for id in instances.split(",")]
        for instance_id in instance_ids:
            try:
                instance_details = api.get_instance(instance_id)
                if instance_details.get("ip"):
                    instance_data.append(
                        {
                            "id": instance_id,
                            "name": instance_details.get("name", "Unknown"),
                            "ip": instance_details["ip"],
                        }
                    )
                else:
                    logger.warning(f"Instance {instance_id} has no IP address")
            except Exception as e:
                logger.error(f"Could not get details for instance {instance_id}: {e}")
    else:
        click.echo("Must specify either --instances or --deployment-file", err=True)
        sys.exit(1)

    if not instance_data:
        click.echo("No valid instances with IP addresses found", err=True)
        sys.exit(1)

    # Extract hostnames/IPs for remote execution
    hostnames = [inst["ip"] for inst in instance_data]

    logger.info(f"Running GPU load tests on {len(hostnames)} instances")
    click.echo(f"Starting GPU load tests on {len(hostnames)} instances...")

    # Create remote executor with monitoring config
    try:
        monitoring_config = MonitoringConfig(
            enabled=monitoring,
            interval_seconds=monitoring_interval,
            duration_buffer_seconds=300.0,
            metrics=["cpu", "memory", "gpu", "network", "disk"]
        )
        
        executor = RemoteLoadTestExecutor(ssh_key, monitoring_config=monitoring_config)

        # Run parallel load tests
        results = executor.run_parallel_load_tests(
            hostnames=hostnames,
            username=username,
            quick=quick,
            max_workers=max_workers,
        )

        # Aggregate results
        aggregated = executor.aggregate_results(results)

        # Save results
        output_file = output or f"gpu_loadtest_results_{int(time.time())}.json"
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = METRICS_DIR / output_file

        with open(results_path, "w") as f:
            json.dump(aggregated.model_dump(), f, indent=2)

        # Print summary
        summary = aggregated.summary
        click.echo(f"\nGPU Load Test Results:")
        click.echo("=" * 50)
        click.echo(f"Total instances: {summary['total_instances']}")
        click.echo(f"Successful tests: {summary['successful_instances']}")
        click.echo(f"Failed tests: {summary['failed_instances']}")
        click.echo(f"Success rate: {summary['success_rate']:.1%}")
        if monitoring:
            click.echo(f"Monitoring enabled: Yes (interval: {monitoring_interval}s)")
        else:
            click.echo(f"Monitoring enabled: No")

        if aggregated.performance_summary:
            perf = aggregated.performance_summary
            click.echo(f"\nPerformance Summary:")
            click.echo(f"Total GPU tests: {perf['total_tests']}")

            if "training" in perf:
                training = perf["training"]
                click.echo(
                    f"Training avg throughput: {training['avg_throughput']:.1f} samples/sec"
                )
                click.echo(
                    f"Training max throughput: {training['max_throughput']:.1f} samples/sec"
                )

            if "inference" in perf:
                inference = perf["inference"]
                click.echo(
                    f"Inference avg throughput: {inference['avg_throughput']:.1f} samples/sec"
                )
                click.echo(
                    f"Inference max throughput: {inference['max_throughput']:.1f} samples/sec"
                )

        # Display resource monitoring summary if available
        if monitoring and aggregated.successful_results:
            monitoring_results = [r for r in aggregated.successful_results if r.monitoring_results]
            if monitoring_results:
                click.echo(f"\nResource Monitoring Summary:")
                for result in monitoring_results:
                    if result.monitoring_results and "snapshots" in result.monitoring_results:
                        snapshots = result.monitoring_results["snapshots"]
                        duration = result.monitoring_results.get("duration_seconds", 0)
                        
                        if snapshots:
                            # Calculate averages from snapshots
                            avg_cpu = sum(s.get("cpu_percent", 0) for s in snapshots) / len(snapshots)
                            avg_memory = sum(s.get("memory_percent", 0) for s in snapshots) / len(snapshots)
                            
                            # Get GPU utilization (first GPU as representative)
                            gpu_utils = []
                            for snapshot in snapshots:
                                if snapshot.get("gpu_metrics") and len(snapshot["gpu_metrics"]) > 0:
                                    gpu_utils.append(snapshot["gpu_metrics"][0].get("utilization_percent", 0))
                            
                            avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
                            
                            click.echo(f"  {result.hostname}: {len(snapshots)} samples over {duration:.1f}s")
                            click.echo(f"    CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%, GPU: {avg_gpu:.1f}%")

        if aggregated.failed_results:
            click.echo(f"\nFailed instances:")
            for failed in aggregated.failed_results:
                click.echo(f"  {failed.hostname}: {failed.error}")

        click.echo(f"\nDetailed results saved to: {results_path}")

    except Exception as e:
        logger.error(f"GPU load test execution failed: {e}")
        click.echo(f"Error running GPU load tests: {e}", err=True)
        sys.exit(1)


@cli.group()
def cleanup():
    """Cleanup commands"""
    pass


@cleanup.command()
@click.option("--instances", help="Comma-separated list of instance IDs to terminate")
@click.option(
    "--deployment-file",
    type=click.Path(exists=True),
    help="Terminate instances from deployment file",
)
@click.confirmation_option(prompt="Are you sure you want to terminate instances?")
@click.pass_context
def terminate(ctx, instances, deployment_file):
    """Terminate instances"""
    logger = ctx.obj["logger"]
    api = get_api_client()

    instance_ids = []

    if deployment_file:
        try:
            with open(deployment_file, "r") as f:
                deployment = json.load(f)
                instance_ids = []
                for inst in deployment.get("instances", []):
                    ids = inst.get("instance_ids", [])
                    if isinstance(ids, list):
                        instance_ids.extend(ids)
                    elif ids:
                        instance_ids.append(ids)
        except Exception as e:
            click.echo(f"Error reading deployment file: {e}", err=True)
            sys.exit(1)
    elif instances:
        instance_ids = [id.strip() for id in instances.split(",")]
    else:
        click.echo("Must specify --instances or --deployment-file", err=True)
        sys.exit(1)

    if not instance_ids:
        click.echo("No instances to terminate")
        return

    logger.info(f"Terminating {len(instance_ids)} instances")

    terminated_count = 0
    for instance_id in instance_ids:
        try:
            result = api.terminate_instance(instance_id)
            logger.info(f"Terminated instance {instance_id}")
            terminated_count += 1
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {e}")

    click.echo(
        f"Successfully terminated {terminated_count}/{len(instance_ids)} instances"
    )


if __name__ == "__main__":
    cli()
