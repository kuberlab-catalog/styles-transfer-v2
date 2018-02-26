import argparse
import json
import logging
import os
import re
import sys
import time

from mlboardclient.api import client

logging.basicConfig(
    format='%(asctime)s %(levelname)-10s %(name)-25s [-] %(message)s',
    level='INFO'
)
SUCCEEDED = 'Succeeded'
FAILED = 'Failed'
LOG = logging.getLogger('RUN_TASK')


def get_parser():
    parser = argparse.ArgumentParser(
        description='Runs specific task with overriding some parameters'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging',
    )
    parser.add_argument(
        '--env',
        '-e',
        help='Pass additional environment variables',
        # nargs='+',
        action='append',
    )
    parser.add_argument(
        '--task',
        '-t',
        required=True,
        help='Task name to start',
    )
    parser.add_argument(
        '--resource-override',
        '-o',
        metavar='<resource>.<param>=<value>',
        help=(
            'Override resource-specific field/parameter.'
        ),
        # nargs='+',
        action='append',
    )
    parser.add_argument(
        '--execution-parameter-override',
        '-p',
        metavar='<resource>.--param=value',
        help='Replace specific execution parameter.',
    )
    parser.add_argument(
        '--build-override',
        '-b',
        action='store_true',
        help='Override $BUILD_ID in all resources execution commands.'
    )
    parser.add_argument(
        '--gpu',
        help='Override GPU requests in all resources.'
    )
    parser.add_argument(
        '--memory',
        help='Override Memory requests in all resources.'
    )
    parser.add_argument(
        '--cpu',
        help='Override CPU requests in all resources.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Apply parameters but don\'t run task actually.'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=0,
        help='Timeout for running task. 0 means no timeout.'
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=3,
        help=(
            'Number in seconds meaning how often to check '
            'task status for completion.'
        )
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.debug:
        logging.root.setLevel('DEBUG')

    ml = client.Client(os.environ.get('MLBOARD_URL'))

    current_task_name = os.environ.get('TASK_NAME')
    LOG.info("Current task name = %s" % current_task_name)

    current_project = os.environ['PROJECT_NAME']
    current_workspace = os.environ['WORKSPACE_ID']
    build_id = os.environ['BUILD_ID']

    LOG.info("Current project = %s" % current_project)
    LOG.info("Current workspace = %s" % current_workspace)

    current_app_id = current_workspace + '-' + current_project
    app = ml.apps.get(current_app_id)

    task = None
    for t in app.tasks:
        if t.name == args.task:
            task = t
            break

    if task is None:
        LOG.error(
            'Task %s not found for project %s.'
            % (args.task, current_project)
        )
        sys.exit(1)

    if args.build_override:
        for r in task.config['resources']:
            r['command'] = r['command'].replace('$BUILD_ID', build_id)
            r['command'] = r['command'].replace('${BUILD_ID}', build_id)
            if r.get('args'):
                r['args'] = r['args'].replace('${BUILD_ID}', build_id)
                r['args'] = r['args'].replace('$BUILD_ID', build_id)

    overrides = args.resource_override or []
    if args.cpu:
        overrides.append('*.resources.requests.cpu=%s' % args.cpu)
        overrides.append('*.resources.limits.cpu=%s' % args.cpu)
    if args.gpu:
        overrides.append('*.resources.requests.gpu=%s' % args.gpu)
        overrides.append('*.resources.limits.gpu=%s' % args.gpu)
        overrides.append('*.resources.accelerators.gpu=%s' % args.gpu)
    if args.memory:
        overrides.append('*.resources.requests.memory=%s' % args.memory)
        overrides.append('*.resources.limits.memory=%s' % args.memory)

    apply_env(task, args.env)
    apply_resource_overrides(task, overrides)
    LOG.debug("TASK JSON:\n%s" % json.dumps(task.config, indent=2))

    if not args.dry_run:
        task = task.start()
        while not task.completed:
            task.refresh()
            LOG.info(task)
            time.sleep(args.check_interval)
        # task.run(timeout=args.timeout, delay=args.check_interval)


def apply_env(task, envs):
    if not envs:
        return

    env_vars = {}
    for e in envs:
        name_value = e.split('=')
        if len(name_value) < 2:
            raise RuntimeError(
                'Invalid env override spec: %s' % e
            )

        name = name_value[0]
        value = '='.join(name_value[1:])
        env_vars[name] = value

    LOG.debug("ENV VARS: %s", env_vars)
    for r in task.config['resources']:
        env_override = env_vars.copy()
        for e in r.get('env', []):
            if e['name'] in env_override:
                e['value'] = env_override.pop(e['name'])

        if not r.get('env'):
            r['env'] = []

        for n, v in env_override.items():
            r['env'].append({'name': n, 'value': v})


def apply_resource_overrides(task, resource_overrides):
    if not resource_overrides:
        return

    LOG.debug("OVERRIDES: %s" % resource_overrides)
    cfg = task.config
    for override in resource_overrides:
        splitted = override.split('=')
        if len(splitted) < 2:
            raise RuntimeError(
                'Invalid resource override spec: %s' % override
            )
        path = splitted[0]
        value = '='.join(splitted[1:])

        splitted = path.split('.')
        if len(splitted) < 2:
            raise RuntimeError(
                'Invalid resource override path: %s' % path
            )
        resource = splitted[0]
        path = splitted[1:-1]

        LOG.debug('resource=%s' % resource)
        LOG.debug('path=%s' % splitted[1:])

        for r in cfg['resources']:
            if r['name'] != resource and resource != '*':
                continue
            if r['name'] == resource or resource == '*':
                to_replace = r
                for p in path:
                    inner = to_replace.get(p)
                    if inner is None:
                        to_replace[p] = {}
                    to_replace = to_replace[p]

                v = to_replace.get(splitted[-1])
                if not v:
                    if value.isdigit():
                        to_replace[splitted[-1]] = int(value)
                    elif re.match("^[\.0-9]+$", value):
                        to_replace[splitted[-1]] = float(value)
                    else:
                        to_replace[splitted[-1]] = value
                    break

                if isinstance(to_replace[splitted[-1]], int):
                    to_replace[splitted[-1]] = int(value)
                elif isinstance(to_replace[splitted[-1]], float):
                    to_replace[splitted[-1]] = float(value)
                else:
                    to_replace[splitted[-1]] = value


if __name__ == '__main__':
    main()
