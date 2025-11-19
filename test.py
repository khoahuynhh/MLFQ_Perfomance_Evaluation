import simpy


def process(env):
    while True:
        print(f"Thời gian: {env.now}")
        yield env.timeout(1)


env = simpy.Environment()
env.process(process(env))
env.run(until=5)
