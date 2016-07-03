import random
import operator
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha = 0.05, gamma = 0.95, epsilon = .2):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default = 0
        self.Q_table = {}
        self.policy = {}
        self.temp_dict = {}
        self.next_state = None
        


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state

        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        
        
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
        
        self.next_state = (self.planner.next_waypoint(), self.env.sense(self)['light'], self.env.sense(self)['oncoming'], self.env.sense(self)['left'])  # agent senses new state after taking action by self.env.act
        
        
        # TODO: Learn policy based on state, action, reward

        self.Q_table[(self.state,action)] = (1-self.alpha)*(self.Q_table.get((self.state,action),self.default)) + self.alpha*(reward + (self.gamma)*(max([self.Q_table.get((self.next_state,None),self.default), self.Q_table.get((self.next_state,'forward'),self.default), self.Q_table.get((self.next_state,'left'),self.default), self.Q_table.get((self.next_state,'right'), self.default)])))    # Updating q-values
        
        self.temp_dict = {}
       
        
        for key in self.Q_table.keys():                      # self.temp_dict = {Q_table[(self.state, action)] | action could be anything} (I am not proud of this part of the code)
            if key[0] == self.state:
                self.temp_dict[key[1]] = self.Q_table[key] 
       
        
        self.policy[self.state] = max(self.temp_dict.iteritems(), key=operator.itemgetter(1))[0]  # Updating policy 


        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.000000000000005, display=False)  # create simulator (uses pygame when display=True, if available)
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
    print "\n\n"

    
   

if __name__ == '__main__':
    run()
