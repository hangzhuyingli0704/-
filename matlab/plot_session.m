% plot_session.m
% MATLAB plotting helper for paper figures.
%
% Usage:
%   T = readtable('data/raw_sessions/realtime_log.csv');
%   plot_session(T);
%
% Or:
%   plot_session('data/raw_sessions/realtime_log.csv');

function plot_session(input)
    if ischar(input) || isstring(input)
        T = readtable(input);
    else
        T = input;
    end

    if any(strcmp('timestamp', T.Properties.VariableNames))
        t = datetime(T.timestamp, 'ConvertFrom', 'posixtime');
    else
        error('timestamp column not found');
    end

    % 1) Eye features
    figure('Name','Eye Features');
    plot(t, T.mean_EAR, '-'); hold on;
    plot(t, T.perclos, '-');
    legend('mean\_EAR','perclos','Location','best');
    xlabel('Time'); ylabel('Value'); grid on;

    % 2) Mouth features
    figure('Name','Mouth Features');
    plot(t, T.mean_MAR, '-'); hold on;
    plot(t, T.MAR_peak, '-');
    stem(t, T.yawn_count, 'filled');
    legend('mean\_MAR','MAR\_peak','yawn\_count','Location','best');
    xlabel('Time'); ylabel('Value'); grid on;

    % 3) Head features
    figure('Name','Head Features');
    plot(t, T.pitch_mean, '-'); hold on;
    plot(t, T.pitch_var, '-');
    stem(t, T.nod_count, 'filled');
    legend('pitch\_mean','pitch\_var','nod\_count','Location','best');
    xlabel('Time'); ylabel('Value'); grid on;

    % 4) Model output (if exists)
    if any(strcmp('pred', T.Properties.VariableNames))
        figure('Name','Fatigue Prediction');
        yyaxis left;
        stairs(t, T.pred, 'LineWidth', 1.5); ylim([-0.2 2.2]);
        yticks([0 1 2]); yticklabels({'LOW','MID','HIGH'});
        ylabel('Predicted fatigue');

        if any(strcmp('p_high', T.Properties.VariableNames))
            yyaxis right;
            plot(t, T.p_high, '-'); ylim([0 1]);
            ylabel('P(high)');
        end

        xlabel('Time'); grid on;
    end

    % 5) If ground-truth label exists, show comparison & confusion
    if any(strcmp('label', T.Properties.VariableNames))
        valid = ~isnan(T.label);
        if any(valid)
            figure('Name','Label vs Prediction');
            stairs(t(valid), T.label(valid), 'LineWidth', 1.5); hold on;
            if any(strcmp('pred', T.Properties.VariableNames))
                stairs(t(valid), T.pred(valid), 'LineWidth', 1.5);
                legend('label','pred','Location','best');
            else
                legend('label','Location','best');
            end
            yticks([0 1 2]); yticklabels({'LOW','MID','HIGH'});
            xlabel('Time'); ylabel('Class'); grid on;

            if any(strcmp('pred', T.Properties.VariableNames))
                ytrue = T.label(valid);
                ypred = T.pred(valid);
                C = confusionmat(ytrue, ypred, 'Order', [0 1 2]);
                disp('Confusion matrix (rows=true, cols=pred; order=0,1,2):');
                disp(C);
            end
        end
    end
end
