import numpy as np
import os
from wireless.envs.time_freq_resource_allocation import TimeFreqResourceAllocationV0
from wireless.agents.time_freq_resource_allocation_v0.round_robin_agent import RoundRobinAgent
from wireless.agents.time_freq_resource_allocation_v0.proportional_fair import ProportionalFairAgent

bw_mhz = 5
num_user = 8

def save_state(env):
    return {
        'cqi':          np.copy(env.cqi),
        'prbs':         np.copy(env.Nf),
        's':            np.copy(env.s),
        'e':            np.copy(env.e),
        'qi':           np.copy(env.qi),
        'tti':          env.tti,
        't':            env.t,
        'tti_next_pkt': np.copy(env.tti_next_pkt),
        'ue_pos':       np.copy(env.ue_pos),
        'ue_v_mps':     np.copy(env.ue_v_mps),
        'ue_dir':       np.copy(env.ue_dir),
    }

def restore_state(env, snapshot):
    env.cqi          = np.copy(snapshot['cqi'])
    env.s            = np.copy(snapshot['s'])
    env.e            = np.copy(snapshot['e'])
    env.qi           = np.copy(snapshot['qi'])
    env.tti          = snapshot['tti']
    env.t            = snapshot['t']
    env.tti_next_pkt = np.copy(snapshot['tti_next_pkt'])
    env.ue_pos       = np.copy(snapshot['ue_pos'])
    env.ue_v_mps     = np.copy(snapshot['ue_v_mps'])
    env.ue_dir       = np.copy(snapshot['ue_dir'])
    env.p            = env.t % env.Nf
    env._recalculate_rf()
    env._update_state()
    return env.state

def run_rollout(env, snapshot, AgentClass, n_steps=8):
    restore_state(env, snapshot)
    agent = AgentClass(env.action_space, env.K, env.L)
    for _ in range(n_steps):
        _, reward_sep, done, _ = env.step(agent.act(env.state, 0, False))
        if done:
            break
        te=save_state(env)
    return save_state(env)

def run_experiment(total_episodes=5000, burn_in=1, rollout=128, buffer_max_size=32, out_dir='data'):
    dataset = {
        "X": [],"Y0_RR": [], "Y1_PFCA": [],
    }

    for episode in range(total_episodes):
        T=1
        n_prbs=np.random.randint(4,12)
        env = TimeFreqResourceAllocationV0(n_ues=num_user, n_prbs=n_prbs, buffer_max_size=buffer_max_size, it=10)
        env.reset()
        env.seed(episode)
        x_snapshot = save_state(env)
        x_backlog = np.sum(x_snapshot['s'], axis=1)
        
        env.seed(episode)
        y0_snapshot = run_rollout(env, x_snapshot, RoundRobinAgent,  n_steps=T*n_prbs)
        y0_backlog = np.sum(y0_snapshot['s'],axis=1)
        env.seed(episode)
        y1_snapshot = run_rollout(env, x_snapshot, ProportionalFairAgent, n_steps=T*n_prbs)
        y1_backlog = np.sum(y1_snapshot['s'],axis=1)
        dataset[f"X"].append(x_snapshot)
        dataset["Y0_RR"].append(y0_backlog)
        dataset["Y1_PFCA"].append(y1_backlog)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"scheduling_data.npy"), dataset,allow_pickle=True)

    print(f"\nDataset saved")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--burn_in',        type=int, default=1)
    parser.add_argument('--rollout',        type=int, default=1)
    parser.add_argument('--buffer',         type=int, default=32)
    parser.add_argument('--n_episodes',     type=int, default=40000)
    parser.add_argument('--out_dir',        type=str, default='data')
    args = parser.parse_args()
    run_experiment(total_episodes=args.n_episodes, burn_in=args.burn_in,
                   rollout=args.rollout, buffer_max_size=args.buffer, out_dir=args.out_dir)