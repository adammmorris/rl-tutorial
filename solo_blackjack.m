%% CODE FOR MISSION 3
% Adam Morris ? Computational Social Cognition Bootcamp, July 2017

%% Simulate

% initialize parameters
numRounds = 100000;
numStates = 44; % all possible sums
numActions = 2; % 1 = hit, 2 = stay

% initialize reward matrix
rewardMatrix = zeros(numStates, 1);
rewardMatrix(10 : 15) = 10;
rewardMatrix(15 : 20) = 20;
rewardMatrix(21) = 100;
rewardMatrix(22 : end) = -50;

% initialize matrix of Q values, other parameters
Q = zeros(numStates, numActions);
alpha = .01; % learning rate
beta = .1; % inverse temperature

% for recording results
% we want to record 3 things: the first choice, second choice, and reward
results = zeros(numRounds, 3);

for curRound = 1:numRounds
   cards = [];
   
   %% Deal initial hand
   cards(end + 1) = getRandomCard();
   cards(end + 1) = getRandomCard(); 
   
   % Make first decision
   state1 = sum(cards);
   actionProbabilities = exp(beta * Q(state1, :)) / sum(exp(beta * Q(state1, :)));
   action1 = randsample(1:numActions, 1, true, actionProbabilities);

   if action1 == 1 % if they hit...
       % get new card
       cards(end + 1) = getRandomCard();
       state2 = sum(cards);
       
       % update, choose again
       Q(state1, action1) = Q(state1, action1) + alpha * (max(Q(state2, :)) - Q(state1, action1));

       actionProbabilities = exp(beta * Q(state2, :)) / sum(exp(beta * Q(state2, :)));
       action2 = randsample(1:numActions, 1, true, actionProbabilities);   

       if action2 == 1 % if they hit..
          cards(end + 1) = getRandomCard(); % get new card
          finalState = sum(cards);
       else % if they stayed, end it
          finalState = state2;
       end
   else
       % If they stayed, end it
       finalState = state1;
   end
   
   reward = rewardMatrix(finalState);
   
   % update the reward to their last move
   if action1 == 2
       Q(state1, action1) = Q(state1, action1) + alpha * (reward - Q(state1, action1));
   elseif action1 == 1
       Q(state2, action2) = Q(state2, action2) + alpha * (reward - Q(state2, action2));
   end
   
   results(curRound, :) = [action1 action2 reward];
end

%% Plot results

plotResults(Q, 4);