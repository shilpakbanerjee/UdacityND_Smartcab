import random
import operator
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, gamma = 0.10):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = 1.0
        self.gamma = gamma
        self.epsilon = 1.0
        self.default = 0
        self.Q_table = {}
        self.policy = {}
        self.temp_dict = {}
        self.next_state = None
        self.cumulative_reward = 0
        self.trial = -1
        self.reward_list = []      # records the cumulative rewards for all 100 trials
        self.penalty_list = []     # records the cumulative penalty for all 100 trials
        self.success_count = []    # records 1 is success and 0 if fail
        self.no_of_moves = []      # records the number of moves when trial ends
        self.temp = 0


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.trial += 1
        self.reward_list.append(0.0)
        self.success_count.append(0)
        self.penalty_list.append(0.0)
        self.no_of_moves.append(0)
    
    def get_decay_rate(self):          #Decay rate for alpha and epsilon
        return 1.0/float(self.trial + 1)
        
                
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
 
        
        if self.trial > 88:
            self.epsilon = self.get_decay_rate()             # decaying alpha and epsilon more exploitation and less exploration with time
            self.alpha = self.get_decay_rate()
            
        

        # TODO: Update state

        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        
        self.no_of_moves[self.trial] = t
        
        # TODO: Select action according to your policy

        
        
        if random.random() > self.epsilon:                                       # epsilon-greedy choice of policy: probability of random action = epsilon
            if self.state in self.policy.keys():                                 # agent may be seeing this state for the first time
                action = self.policy[self.state]
            else:
                action = random.choice([None, 'forward', 'left', 'right'])
        else:
            action = random.choice([None, 'forward', 'left', 'right'])
            
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        self.reward_list[self.trial] += reward

        if reward < 0:
            self.penalty_list[self.trial] += reward
        
        
        
        self.next_state = (self.planner.next_waypoint(), self.env.sense(self)['light'], self.env.sense(self)['oncoming'], self.env.sense(self)['left'])  # agent senses new state after taking action by self.env.act
        
        
        # TODO: Learn policy based on state, action, reward

        self.Q_table[(self.state,action)] = (1.0 - self.alpha)*(self.Q_table.get((self.state,action),self.default)) + self.alpha*(reward + (self.gamma)*(max([self.Q_table.get((self.next_state,None),self.default), self.Q_table.get((self.next_state,'forward'),self.default), self.Q_table.get((self.next_state,'left'),self.default), self.Q_table.get((self.next_state,'right'), self.default)])))    # Updating q-values
        
        self.temp_dict = {}
       
        
        for key in self.Q_table.keys():                      # self.temp_dict = {Q_table[(self.state, action)] | action could be anything} (I am not proud of this part of the code)
            if key[0] == self.state:
                self.temp_dict[key[1]] = self.Q_table[key] 
       
        
        self.policy[self.state] = max(self.temp_dict.iteritems(), key=operator.itemgetter(1))[0]  # Updating policy 


        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        if self.env.done == True:
            self.success_count[self.trial] = 1


        
         
        
        


def run():
    """Run the agent for a finite number of trials."""

    rewards_10 = 0.0
    success_10 = 0
    penalty_10 = 0.0
    moves_10 = 0
    n_runs = 1         # MUST SET THIS = 1 BEFORE UN COMMENTING ANYTHING IN THE FOR LOOK OR SETTING DISPLAY = TRUE. SET THIS = 10 OR LARGER FOR GOOD ESTIMATE OF PERFORMANCE                   
  
    for i in range(0,n_runs):                                    # Set range to range(0,1) for this algorithm to run the intended way
        # Set up environment and agent
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        e.set_primary_agent(a, enforce_deadline= True)  # specify agent to track
        # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

        # Now simulate it
        sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
        # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    
        sim.run(n_trials=100)  # run for a specified number of trials
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    


        # Uncomment to see final policy learned by the agent

        print "\n\n\nFinal policy learned by agent:\n" 
        print "{:<40} {:<20}\n".format('(STATE)','ACTION')
        for k in sorted(a.policy):
            print "{:<40} {:<20}".format(k,a.policy[k])
    
        # Uncomment to see final set of q-values learned by the agent

        print "\n\n\nFinal list of Q values:\n" 
        print "{:<50} {:<20}\n".format('((STATE),ACTION)','Q VALUE')
        for k in sorted(a.Q_table):
            print "{:<50} {:<20}".format(k,a.Q_table[k])

        rewards_10 += np.mean(a.reward_list[-10:])
        penalty_10 += np.mean(a.penalty_list[-10:])
        moves_10 += np.mean(a.no_of_moves[-10:])
        success_10 += np.sum(a.success_count[-10:])
        print(a.success_count[-10:])
        
    
    print "\n\n\n"
    print "="*133
    print "Following numbers represent the average of 10 runs of the entire algorithm with parameter: gamma = {:<5}".format(a.gamma)
    print "="*133
    print "Average reward for the last 10 trips of the smartcab: {:<10}\n".format(rewards_10/float(n_runs))
    print "Average penalty for the last 10 trips of the smartcab: {:<10}\n".format(-penalty_10/float(n_runs))
    print "Average number of moves for the last 10 trips of the smartcab: {:<10}\n".format(moves_10/float(n_runs))
    print "Average number of successfully completed trips for the last 10 trips of the smartcab: {:<10}".format(success_10/float(n_runs))
    print "="*133
    print "\n\n\n"

if __name__ == '__main__':
    run()
