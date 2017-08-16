%% CODE FOR MISSION 1
% Adam Morris ? Computational Social Cognition Bootcamp, July 2017

%% Simulate

% initialize parameters
numRounds = 100;
numStates = 3;
numActions = 2;

% initialize transition matrix
transitionMatrix = zeros(numStates, numActions, numStates);
transitionMatrix(1, 1, 2) = 1; % in state 1, taking action 1 leads you (with probability 1) to state 2
transitionMatrix(1, 2, 3) = 1; % in state 1, taking action 2 leads you (with probability 1) to state 3

% initialize reward matrix
rewardMatrix = zeros(numStates, numActions, numStates);
rewardMatrix(1, 1, 2) = -1; % left arm (action 1) gives you -1
rewardMatrix(1, 2, 3) = 1; % right arm (action 2) gives you +1

% initialize matrix of Q values, other parameters
Q = zeros(numStates, numActions);
alpha = .01; % learning rate
beta = 10; % inverse temperature

% for recording results
% we want to record 2 things: the choice and reward
results = zeros(numRounds, 2);

for curRound = 1:numRounds
   % start in state 1
   state = 1;
   
   % make decision with softmax function
   actionProbabilities = exp(beta * Q(state, :)) / sum(exp(beta * Q(state, :)));
   action = randsample(1:numActions, 1, true, actionProbabilities);
   
   % what happened?
   nextStateProbabilities = transitionMatrix(state, action, :);
   nextState = randsample(1:numStates, 1, true, nextStateProbabilities);
   
   %reward = rewardMatrix(state, action, nextState);
   % if doing stochastic rewards, uncomment the next line
   reward = rewardMatrix(state, action, nextState) + randn() * .1;
   
   % learn!
   Q(state, action) = Q(state, action) + alpha * (reward - Q(state, action));
   
   % record
   results(curRound, :) = [action reward];
end

%% Plot results

plotResults(results, 1);