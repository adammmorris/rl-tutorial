%% CODE FOR MISSION 4
% Adam Morris ? Computational Social Cognition Bootcamp, July 2017

% initialize parameters
numRounds = 100000;
numPlayerStates = 44; % 21 possible sums, plus "bust"
numDealerStates = 44;
numActions = 2; % 1 = hit, 2 = stay

% initialize reward values
reward_win = 1;
reward_lose = 0;

% initialize matrix of Q values, other parameters
% Note the two-dimensional state space!!
Q = zeros(numPlayerStates, numDealerStates, numActions);
alpha = .01; % learning rate
beta = .1; % inverse temperature

dealer_shows_card = true;

% for recording results
% we want to record 3 things: the first choice, second choice, and reward
results = zeros(numRounds, 3);

for curRound = 1:numRounds
   playerCards = [];
   dealerCards = [];
   
   %% Deal initial hand
   playerCards(end + 1) = getRandomCard();
   playerCards(end + 1) = getRandomCard(); 
   dealerCards(end + 1) = getRandomCard();
   dealerCards(end + 1) = getRandomCard(); 
   
   %% Make first decision
   playerState1 = sum(playerCards);
   
   if dealer_shows_card
       dealerState = dealerCards(1); % player observes a card
   else
       dealerState = 1; % player observes nothing
   end
   
   actionProbabilities = exp(beta * Q(playerState1, dealerState, :)) / sum(exp(beta * Q(playerState1, dealerState, :)));
   action1 = randsample(1:numActions, 1, true, actionProbabilities);

   if action1 == 1 % if they hit...
       % get new card
       playerCards(end + 1) = getRandomCard();
       playerState2 = sum(playerCards);
       
       % update, choose again
       Q(playerState1, dealerState, action1) = Q(playerState1, dealerState, action1) + ...
           alpha * (max(Q(playerState2, dealerState, :)) - Q(playerState1, dealerState, action1));

       actionProbabilities = exp(beta * Q(playerState2, dealerState, :)) / sum(exp(beta * Q(playerState2, dealerState, :)));
       action2 = randsample(1:numActions, 1, true, actionProbabilities);

       if action2 == 1 % if they hit..
          playerCards(end + 1) = getRandomCard(); % get new card
          finalPlayerState = sum(playerCards);
       else % if they stayed, end it
          finalPlayerState = playerState2;
       end
   else
       % If they stayed, end it
       finalPlayerState = playerState1;
   end
   
   %% Determine reward
   
   % dealer reveals their other card
   newDealerState = sum(dealerCards);
   
   % did the player bust, or did the dealer get a blackjack?
   if finalPlayerState > 21 || newDealerState == 21
       reward = reward_lose; % player loses
   else % otherwise, dealer continues
       while newDealerState < 17 
           % while dealer has < 17, they hit
           dealerCards(end + 1) = getRandomCard();
           newDealerState = sum(dealerCards);
       end
       
       % did player win?
       if finalPlayerState > newDealerState
           reward = reward_win;
       else
           reward = reward_lose;
       end
   end
   
   %% Final update
   
   % update reward to last action they took
   % this is tricky - we have to update based on the dealer state that we
   % knew before, not the one we know now
   % because that's what we should be learning about
   if action1 == 2
       Q(playerState1, dealerState, action1) = Q(playerState1, dealerState, action1) + ...
           alpha * (reward - Q(playerState1, dealerState, action1));
   else
       Q(playerState2, dealerState, action2) = Q(playerState2, dealerState, action2) + ...
           alpha * (reward - Q(playerState2, dealerState, action2));
   end

   results(curRound, :) = [action1 action2 reward];
end

%% Plot results

if dealer_shows_card
    plotResults(Q, 4);
else
    plotResults(squeeze(Q(:, 1, :)), 4);
end