% plot_feature_importance.m
% Plot feature importance from *_feature_importance.csv (produced by train_model.py).
%
% Usage:
%   plot_feature_importance('models/fatigue_model_feature_importance.csv');

function plot_feature_importance(csv_path)
    T = readtable(csv_path);
    [imp_sorted, idx] = sort(T.importance, 'descend');
    names_sorted = T.feature(idx);

    figure('Name','Feature Importance');
    bar(imp_sorted);
    set(gca,'XTick',1:numel(names_sorted),'XTickLabel',names_sorted,'XTickLabelRotation',45);
    ylabel('Importance'); grid on;
end
