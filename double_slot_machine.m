%% CODE FOR MISSION 2
% Adam Morris ? Computational Social Cognition Bootcamp, July 2017

%% Simulate

% initialize parameters
numRounds = 500;
numStates = 7;
numActions = 2;

% initialize transition matrix
transitionMatrix = zeros(numStates, numActions, numStates);
stochastic_transitions = true;

if stochastic_transitions
    transitionMatrix(1, 1, 2) = .7; % in state 1, taking action 1 leads you (with probability .7) to state 2
    transitionMatrix(1, 1, 3) = .3; % in state 1, taking action 1 leads you (with probability .3) to state 3
    transitionMatrix(1, 2, 2) = .3; % in state 1, taking action 2 leads you (with probability .3) to state 2
    transitionMatrix(1, 2, 3) = .7; % in state 1, taking action 2 leads you (with probability .7) to state 3
else
    transitionMatrix(1, 1, 2) = 1; % in state 1, taking action 1 leads you (with probability 1) to state 2
    transitionMatrix(1, 2, 3) = 1; % in state 1, taking action 2 leads you (with probability 1) to state 3
end

transitionMatrix(2, 1, 4) = 1; % in state 2, taking action 1 leads you (with probability 1) to state 4
transitionMatrix(2, 2, 5) = 1; % in state 2, taking action 2 leads you (with probability 1) to state 5
transitionMatrix(3, 1, 6) = 1; % in state 3, taking action 1 leads you (with probability 1) to state 6
transitionMatrix(3, 2, 7) = 1; % in state 3, taking action 2 leads you (with probability 1) to state 7

% initialize reward matrix
rewardMatrix = zeros(numStates, numActions, numStates);
rewardMatrix(2, 1, 4) = -1; % in left slot, left arm (action 1) gives you -1
rewardMatrix(2, 2, 5) = 1; % in left slot, right arm (action 2) gives you +1
rewardMatrix(3, 1, 6) = -2; % in right slot, left arm (action 1) gives you -1
rewardMatrix(3, 2, 7) = 2; % in right slot, right arm (action 2) gives you +2

% initialize matrix of Q values, other parameters
Q = zeros(numStates, numActions);
alpha = .01; % learning rate
beta = 10; % inverse temperature

% for recording results
% we want to record 3 things: the first choice, second choice, and reward
results = zeros(numRounds, 3);

for curRound = 1:numRounds
   %% First decision 
   
   % start in state 1
   state1 = 1;
   
   % make first decision with softmax function
   actionProbabilities = exp(beta * Q(state1, :)) / sum(exp(beta * Q(state1, :)));
   action1 = randsample(1:numActions, 1, true, actionProbabilities);
   
   % what happened?
   nextStateProbabilities = transitionMatrix(state1, action1, :);
   nextState = randsample(1:numStates, 1, true, nextStateProbabilities);
   reward = rewardMatrix(state1, action1, nextState);
   
   % learn!
   Q(state1, action1) = Q(state1, action1) + alpha * (reward + max(Q(nextState, :)) - Q(state1, action1));
   
   %% Second decision
   state2 = nextState; % we moved to the next state!
   
   % make second decision
   actionProbabilities = exp(beta * Q(state2, :)) / sum(exp(beta * Q(state2, :)));
   action2 = randsample(1:numActions, 1, true, actionProbabilities);
   
   % what happened?
   nextStateProbabilities = transitionMatrix(state2, action2, :);
   nextState = randsample(1:numStates, 1, true, nextStateProbabilities);
   reward = rewardMatrix(state2, action2, nextState);
   
   % learn!
   Q(state2, action2) = Q(state2, action2) + alpha * (reward - Q(state2, action2));
   
   % record
   results(curRound, :) = [action1 action2 reward];
end

%% Plot results

plotResults(results, 2);