%% Plot results for all missions
% Adam Morris ? Computational Social Cognition Bootcamp, July 2017

function plotResults(results, tasknum)

if tasknum == 1 % Task 1
    numRounds = size(results, 1);

    % Partition game into bins of 10 rounds each
    divisor = 10;
    partitions = 1:divisor:numRounds;
    numPartitions = length(partitions);

    avgReward = zeros(numPartitions, 1);
    avgChoice = zeros(numPartitions, 1);
    
    for curPartition = 1:numPartitions
        if curPartition < numPartitions
            range = partitions(curPartition) : partitions(curPartition + 1);
        else
            range = partitions(curPartition) : numRounds;
        end

        avgReward(curPartition) = mean(results(range, 2));
        avgChoice(curPartition) = mean(results(range, 1) == 2);
    end

    xrange = (1 : numPartitions) * divisor;

    % Plot actions & rewards
    figure

    subplot(1, 2, 1);
    plot(xrange, avgChoice, 'LineWidth', 4);
    ylim([0 1.1]);
    title('Choices');
    xlabel('Round number');
    ylabel('Prob(Action 2)');
    set(gca, 'LineWidth', 2);
    set(gca, 'FontSize', 36);
    
    subplot(1, 2, 2);
    plot(xrange, avgReward, 'LineWidth', 4, 'Color', 'red');
    ylim([-1 1.2]);
    title('Average reward');
    xlabel('Round number');
    ylabel('Reward');
    set(gca, 'LineWidth', 2);
    set(gca, 'FontSize', 36);

elseif tasknum == 2 % Task 2 - same thing as before, but plot both first & second choices
    numRounds = size(results, 1);
    divisor = 10;

    partitions = 1:divisor:numRounds;
    numPartitions = length(partitions);

    avgReward = zeros(numPartitions, 1);
    avgChoice = zeros(numPartitions, 1);
    avgChoice2 = zeros(numPartitions, 1);
    
    for curPartition = 1:numPartitions
        if curPartition < numPartitions
            range = partitions(curPartition) : partitions(curPartition + 1);
        else
            range = partitions(curPartition) : numRounds;
        end

        avgReward(curPartition) = mean(results(range, 3));
        avgChoice(curPartition) = mean(results(range, 1) == 2);
        avgChoice2(curPartition) = mean(results(range, 2) == 2);
    end

    xrange = (1 : numPartitions) * divisor;

    figure

    subplot(1, 3, 1);
    plot(xrange, avgChoice, 'LineWidth', 4);
    ylim([0 1.1]);
    title('Choice 1');
    xlabel('Round number');
    ylabel('Prob(Action 2)');
    set(gca, 'LineWidth', 2);
    set(gca, 'FontSize', 36);
    
    subplot(1, 3, 2);
    plot(xrange, avgChoice2, 'LineWidth', 4);
    ylim([0 1.1]);
    title('Choice 2');
    xlabel('Round number');
    ylabel('Prob(Action 2)');
    set(gca, 'LineWidth', 2);
    set(gca, 'FontSize', 36);
    
    subplot(1, 3, 3);
    plot(xrange, avgReward, 'LineWidth', 4, 'Color', 'red');
    ylim([-2 2]);
    title('Average reward');
    xlabel('Round number');
    ylabel('Reward');
    set(gca, 'LineWidth', 2);
    set(gca, 'FontSize', 36);
    
elseif tasknum == 3 || (tasknum == 4 && ismatrix(results)) % Task 3 & 4A - plot final Q values
    plot(results(4:21, :))
    set(gca, 'XTick', [1, 11-4, 21-4], 'XTickLabel', {'4', '11', '21'});
    legend('Hit', 'Stay')
    title('What has the agent learned?');
    xlabel('Current sum');
    ylabel('Q value');
    set(gca, 'LineWidth', 2);
    set(gca, 'FontSize', 36);
    
elseif tasknum == 4 % Task 4B - plot final Q values for 2D state space
    if any(any(any(results(4:21, 12:21, :) ~= 0)))
        bar3(results(4:21, 4:21, 1) - results(4:21, 4:21, 2))
        set(gca, 'XTick', [1, 17-4, 21-4], 'XTickLabel', {'4', '17', '21'});
    else
        bar3(results(4:21, 2:11, 1) - results(4:21, 2:11, 2))
        set(gca, 'XTick', [1, 9], 'XTickLabel', {'2', '11'});
    end
    title('What has the agent learned?');
    ylabel('Current player sum');
    xlabel('Dealer sum showing');
    zlabel('Relative value of hit > stay');
    set(gca, 'LineWidth', 2);
    set(gca, 'FontSize', 36);
    set(gca, 'YTick', [1, 11-4, 21-4], 'YTickLabel', {'4', '11', '21'});
end