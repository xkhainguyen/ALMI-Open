
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import math
import time
import joblib
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from termcolor import colored
import line_profiler

class ConeGeometry(gymutil.LineGeometry):
    def __init__(self, scale=0.5, pose=None):
        verts = np.empty((501, 2), gymapi.Vec3.dtype)
        verts[0][0] = (0, 0, 0)
        verts[0][1] = (0, 0, -2*scale)
        radius = 0.1 * scale*math.sin(30)
        z = -0.2
        vstep = 2 * math.pi / 500
        idx = 1
        v = 0
        for i in range(500):
            x = radius * math.sin(v)
            y = radius * math.cos(v)
            verts[idx][0] = (0,0,0)
            verts[idx][1] = (x,y,z)
            idx += 1
            v += vstep

        verts[50][0] = (0,0.01,-0.03)
        verts[50][1] = (0,0.01,-2*scale)
        verts[100][0] = (0,-0.01,-0.03)
        verts[100][1] = (0,-0.01, -2*scale)
        verts[150][0] = (0.01,0,-0.03)
        verts[150][1] = (0.01,0,-2*scale)
        verts[200][0] = (-0.01,0,-0.03)
        verts[200][1] = (-0.01,0, -2*scale)
        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        colors = np.empty(501, gymapi.Vec3.dtype)
        colors[:] = (1.0, 0.0, 0.0)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors

class H1_2_WholeBody(LeggedRobot):
    # def __init__(self, cfg: H1_2RoughCfg, sim_params, physics_engine, sim_device, headless):
    #     super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
   
    def add_external_force(self, scale, axis=0):
        # axis 0:x 1:y 2:z
        self.gym.clear_lines(self.viewer)
        forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        forces[:, 0, axis] = scale
        force_positions = self.rb_positions.clone()
        
        force_offset = 0.0
        force_positions[:,0,1] += force_offset
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)

        # TODO visualize force vector
        arrow_geom = ConeGeometry()
        x = force_positions[0,0,0]
        y = force_positions[0,0,1] - 0.1
        z = force_positions[0,0,2]
        pos = gymapi.Transform(gymapi.Vec3(x,y,z), gymapi.Quat.from_euler_zyx(-0.5 * math.pi, 0.5 * math.pi, 0))
        gymutil.draw_lines(arrow_geom, self.gym, self.viewer, self.envs[0], pos)
    
    @line_profiler.profile
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
            
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        actions = self.actions.clone()
        
        if self.cfg.domain_rand.action_delay:
            self.action_history[:, 1:] = self.action_history[:, :-1].clone()
            self.action_history[:, 0] = actions.clone()
            actions = self.action_history[torch.arange(self.num_envs), self.action_delay_steps].clone()
            
        self.render()
        self.last_torques = self.torques
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            # log
            self.target_q_list.append((self.actions[0,:]*0.25+self.default_dof_pos[0,:12]).detach().cpu().numpy())
            self.q_list.append(self.dof_pos[0,:].detach().cpu().clone().numpy())
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    @line_profiler.profile
    def _compute_torques(self, actions):
        control_type = self.cfg.control.control_type
        assert control_type=="P"

        actions_scaled = actions * self.cfg.control.action_scale
        # arm_action = torch.zeros((self.num_envs, 9), dtype=torch.float, device=self.device, requires_grad=False)
        arm_action = (self.motion_buffer[torch.arange(self.num_envs), self.episode_length_buf] - self.default_dof_pos[:, -9:]) * (self.arm_weight if self.cfg.asset.arm_curriculum else 1)
        arm_action[:, 0] = 0 # set torso = 0 
        wholebody_action = torch.cat([actions_scaled, arm_action], dim=-1)
        
        torques = self.p_gains*(wholebody_action + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    @line_profiler.profile
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
            
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
        # # avoid updating command curriculum at each step since the maximum command is common to all envs
            self._update_terrain_curriculum(env_ids)
<<<<<<< HEAD
            # print("terrain update")   
=======
            print("terrain update")   
>>>>>>> 48278cfe2af9586269563fe95574e4fb4a9d3eeb
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)
                 
        
        # update arm_weight curriculum
        if self.cfg.asset.arm_curriculum:
            self.update_arm_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reload motion
        # if self.cfg.asset.arm_curriculum:
        self._alloc_motion(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        if self.add_history:
            self.history_traj[env_ids] *= 0 ####
            self.priv_history_traj[env_ids] *= 0
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.asset.arm_curriculum:
            self.extras["episode"]["arm_curriculum"] = self.arm_weight 
            self.extras["episode"]["motion_curriculum"] = self.motion_weight 
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        
        
        if self.cfg.domain_rand.action_delay:
            self.action_history[env_ids] *= 0.
            self.action_history[env_ids] = 0.
            self.action_delay_steps[env_ids] = torch.randint(self.cfg.domain_rand.action_delay_range[0], 
                                              self.cfg.domain_rand.action_delay_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)
    


    @line_profiler.profile
    def cal_motion_and_arm_weight(self, episode_length_ratio):
        leading_weight = self.motion_weight if self.motion_leading else self.arm_weight
        other_weight = self.arm_weight if self.motion_leading else self.motion_weight
        
        leading_increase = self.motion_weight_increase if self.motion_leading else self.arm_weight_increase
        leading_decrease = self.motion_weight_decrease if self.motion_leading else self.arm_weight_decrease
        other_increase = self.arm_weight_increase if self.motion_leading else self.motion_weight_increase
        other_decrease = self.arm_weight_decrease if self.motion_leading else self.motion_weight_decrease
        
        if episode_length_ratio > 0.8:

            other_weight += other_increase # 0.01
        else:
            other_weight -= other_decrease # 0.1
        
        if other_weight > 1:

            other_weight = 0
            leading_weight += leading_increase # 0.05
            if leading_weight > 1: 
                leading_weight = 1
                other_weight = 1
        if other_weight < 0: 

            other_weight = 1 + other_weight
            leading_weight -= leading_decrease
            if leading_weight < 0: 
                leading_weight = 0
                other_weight = 0
        return leading_weight, other_weight        

    @line_profiler.profile
    def update_arm_curriculum(self, env_ids):

        mean_episode_length = torch.sum(self.episode_length_buf[env_ids]) / len(env_ids)
        
        episode_length_ratio = mean_episode_length / self.max_episode_length


        leading_weight, other_weight = self.cal_motion_and_arm_weight(episode_length_ratio)
        
        self.motion_weight = leading_weight if self.motion_leading else other_weight
        self.arm_weight = other_weight if self.motion_leading else leading_weight


    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:30] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[30:51] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[51:63] = 0. # previous actions
        noise_vec[63:65] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)



        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
 

        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]


        
        self.rb_positions = self.rigid_body_states[:, 0:3].view(self.num_envs, -1, 3)
        
    def _init_buffers(self):


        super()._init_buffers()
        self._init_foot()
        self.last_torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        
        self.arm_weight = self.cfg.asset.init_arm_weight 

        self.add_history = self.cfg.env.add_history
        if self.add_history:

            self.history_length = self.cfg.env.history_length
            self.history_traj = torch.zeros(self.num_envs, self.cfg.env.single_step_obs * self.history_length, device=self.device)
            self.priv_history_traj = torch.zeros(self.num_envs, self.cfg.env.single_step_obs_priv * self.history_length, device=self.device)

        if self.cfg.asset.arm_curriculum:
            self.motion_weight = self.cfg.asset.init_motion_weight
            self.motion_leading = self.cfg.asset.motion_leading

            self.motion_weight_increase = self.cfg.asset.motion_weight_increase
            self.motion_weight_decrease = self.cfg.asset.motion_weight_decrease
            self.arm_weight_increase = self.cfg.asset.arm_weight_increase
            self.arm_weight_decrease = self.cfg.asset.arm_weight_decrease
            self.motion_range = self.cfg.asset.motion_range

        self._load_motion()
        
        if self.cfg.domain_rand.action_delay:
            # store history action
            self.action_history = torch.zeros(self.num_envs, self.cfg.domain_rand.action_delay_range[1]+1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            # action delay step for each env
            self.action_delay_steps = torch.randint(self.cfg.domain_rand.action_delay_range[0], 
                                              self.cfg.domain_rand.action_delay_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)
    

    

    @line_profiler.profile
    def _alloc_motion(self, env_ids):
        if self.cfg.asset.arm_curriculum:
            
            random_lower = max(0, int( (self.motion_weight - self.motion_range) * len(self.new_motion) )) 
            random_upper = min(len(self.new_motion)-1, int( (self.motion_weight + self.motion_range) * len(self.new_motion) ))


            random_indices = torch.randint(random_lower, random_upper+1, (len(env_ids),), device=self.device)

            for i in range(len(env_ids)):

                self.motion_buffer[int(env_ids[i])] = torch.stack(self.new_motion[random_indices[i]][:self.motion_length])
                self.env_motion_dict[int(env_ids[i])] = random_indices[i]

        else: 
            random_indices = torch.randint(0, len(self.new_motion), (self.num_envs,), device=self.device)

            for i in range(len(env_ids)):
                self.motion_buffer[int(env_ids[i])] = torch.stack(self.new_motion[random_indices[i]][:self.motion_length])
                self.env_motion_dict[int(env_ids[i])] = random_indices[i]


    @line_profiler.profile
    def _load_motion(self):
        motion_path = self.cfg.asset.motion_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        print(colored(f"Load motion from {motion_path}......", "green"), end="")
        motion_data = joblib.load(motion_path) # shape: (num_motion, num_frames, num_dof)

        mean_episode_length_path = self.cfg.asset.mean_episode_length_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        import pandas as pd
        df = pd.read_csv(mean_episode_length_path)
        sorted_indices = df.sort_values("mean_episode_length", ascending=False)["env_id"].tolist()
        motion_data = [motion_data[i] for i in sorted_indices] 




        self.motion_data = motion_data 

        new_motion = []
        self.env_motion_dict = {} 
        


        for i in range(len(motion_data)):
            if len(motion_data[i]) < 1002:

                last_row = motion_data[i][-1][:, -9:]
                padding = [torch.from_numpy(last_row).squeeze() for _ in range(1002 - len(motion_data[i]))]

                repeated_frames = []
                for motion_frame in motion_data[i]:
                    frame = torch.from_numpy(motion_frame[:, -9:]).squeeze()
                    repeated_frames.extend([frame, frame])  

                repeated_padding = []
                for pad in padding:
                    repeated_padding.extend([pad, pad])
                new_motion.append(repeated_frames + repeated_padding)
            else:
              
                repeated_frames = []
                for motion_frame in motion_data[i][:1002]:
                    frame = torch.from_numpy(motion_frame[:, -9:]).squeeze()

                new_motion.append(repeated_frames)



 
                   
        self.new_motion = new_motion
        
        # shape: (num_envs, max_episode_length+1, num_arm_dof) 
        motion_length = int(self.max_episode_length+1) 
        self.motion_length = motion_length
        self.motion_buffer = torch.zeros((self.num_envs, motion_length, 9), dtype=torch.float, device=self.device, requires_grad=False)
        

        temp_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # if self.cfg.asset.arm_curriculum:
        self._alloc_motion(temp_env_ids)
        

        
    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def cumpute_upper_body_CoM(self):
        upper_body_name_list = ['torso_link', 
                    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_pitch_link', 
                    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_pitch_link']
        upper_body_indices = [self._body_list.index(name) for name in upper_body_name_list]
        # print(upper_body_indices)

        p_global = self.rigid_body_states_view[:, upper_body_indices, :3]
        q_global = self.rigid_body_states_view[:, upper_body_indices, 3:7]
        c_local = torch.tensor([[self.body_props[i].com.x,self.body_props[i].com.y,self.body_props[i].com.z] for i in upper_body_indices], dtype=torch.float32)  # (N, 3)
        # c_local = torch.tensor([[0,0,0] for i in upper_body_indices], dtype=torch.float32)  # (N, 3)
        c_local = c_local.unsqueeze(0).expand(self.num_envs, -1, -1).to(self.device)
        # print(q_global.shape, c_local.shape)
        c_global = p_global + quat_apply(q_global, c_local)
        # print(p_global)
        # # print(q_global)
        # print(c_global)

        mass_weight = torch.tensor( [ self.body_props[i].mass for i in upper_body_indices] , dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(self.num_envs, -1, 1).to(self.device)
        # print(c_global.shape)
        # print(mass_weight.shape)
        weighted_c_global = c_global * mass_weight
        CoM = weighted_c_global.sum(dim=1) / mass_weight.sum(dim=1)
        # print(CoM)
        # print(self.rigid_body_states_view[:, 0, :3])
        # print(self.rigid_body_states_view[:, 13, :3])
        # print(self.root_states[:, 0:3])
        # print(self.torso_index)
        # print(CoM - self.rigid_body_states_view[:, self.torso_index, :3])
        # print(CoM - self.rigid_body_states_view[:, 0, :3])


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.cumpute_upper_body_CoM()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
    
    def _post_physics_step_callback(self):
        self.update_feet_state()

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), self.cfg.commands.ranges.ang_vel_yaw[0], self.cfg.commands.ranges.ang_vel_yaw[1])

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()


        period = 0.8
        self.is_stance_threshold = 0.55
        self.offset = torch.where(torch.norm(self.commands[:, :3], dim=1) < 0.1, 0, 0.5)  
        

        self.phase = torch.where(torch.norm(self.commands[:, :3], dim=1) < 0.1, 0, (self.episode_length_buf * self.dt) % period / period)
        self.phase_left = self.phase
        self.phase_right = (self.phase + self.offset) % 1
        
        self.left_sin_phase = torch.sin(2 * np.pi * self.phase_left).unsqueeze(1)
        self.right_sin_phase = torch.sin(2 * np.pi * self.phase_right).unsqueeze(1)
        
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
    
    def compute_observations(self):
        """ Computes observations
        """
        
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.left_sin_phase,
                                    self.right_sin_phase
                                    ),dim=-1)
        if self.add_history: 
            self.obs_buf = torch.cat((self.obs_buf, self.history_traj), dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.left_sin_phase,
                                    self.right_sin_phase
                                    ),dim=-1)
        if self.add_history:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.priv_history_traj), dim=-1)

            self.history_traj = torch.cat((self.obs_buf[:, :6], self.obs_buf[:, 9:63] ,self.history_traj[:, 0: (self.history_length-1)*self.single_step_obs]), dim=1)
            self.priv_history_traj = torch.cat((self.privileged_obs_buf[:, :9], self.privileged_obs_buf[:, 12:66], self.priv_history_traj[:, 0: (self.history_length-1)*self.single_step_obs_priv]), dim=1)      

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _process_rigid_body_props(self, props, env_id):

        if env_id==0:
            sum = 0
            left_sum = 0
            right_sum = 0
            other_sum = 0
            for i, (name ,p) in enumerate(zip(self.body_names, props)):
                sum += p.mass

                print(f"Mass of body {i} {name}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")

        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        
        if self.cfg.domain_rand.randomize_base_com:
            torso_index = self._body_list.index("torso_link")
            assert torso_index != -1

            com_x_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.x[0], self.cfg.domain_rand.base_com_range.x[1])
            com_y_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.y[0], self.cfg.domain_rand.base_com_range.y[1])
            com_z_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.z[0], self.cfg.domain_rand.base_com_range.z[1])


        # randomize link mass
        if self.cfg.domain_rand.randomize_link_mass:
            for i, body_name in enumerate(self.cfg.domain_rand.randomize_link_body_names):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                mass_scale = np.random.uniform(self.cfg.domain_rand.link_mass_range[0], self.cfg.domain_rand.link_mass_range[1])
                props[body_index].mass *= mass_scale
                

        return props
    
    def _create_envs(self):

        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names = body_names
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
        torso_name = self.cfg.asset.torso_name
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self._body_list = self.gym.get_actor_rigid_body_names(env_handle, actor_handle)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.body_props = body_props
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
        
        self.torso_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], torso_name)
        
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
    
    # original
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    

    
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques[:,:12]), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:,:12]), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:,:12] / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits[:,:12], dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # print('ang_error:', ang_vel_error[0])
        # print('ang reward:', torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)[0])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        # print('contact force:', torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1))
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    

    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        
        target_feet_height = self.cfg.rewards.feet_swing_height_threshold # old 0.08 
        # target_feet_height = 0.1
        pos_error = torch.square(self.feet_pos[:, :, 2] - target_feet_height) * ~contact 
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[0,2,6,8]]), dim=1)

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_body_states_view[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
    
    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        knee_pos = self.rigid_body_states_view[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.)
        d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_torques_smooth(self):
        return torch.sum(torch.square((self.last_torques[:, :12] - self.torques[:, :12])), dim=1)

    def _reward_stand_still(self): 
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos[:, :12] - self.default_dof_pos[:, :12]), dim=1) * (
                    torch.norm(self.commands[:, :3], dim=1) < 0.1)
    
    def _reward_action_magnitude(self):
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_ankle_torque(self):
        # Penalize ankle torques
        return torch.sum(torch.square(self.torques[:,[4,5,10,11]]), dim=1)

    def _reward_ankle_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,[4,5,10,11]] - self.actions[:,[4,5,10,11]]), dim=1)

    def _reward_stance_base_vel(self):
        # Penalize base velocity when stance
        r = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.1)
        return r
