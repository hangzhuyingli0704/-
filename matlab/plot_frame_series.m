% plot_frame_series.m
% Plot per-frame EAR/MAR/pitch logs.
%
% Usage:
%   plot_frame_series('data/raw_sessions/session_frame.csv');

function plot_frame_series(csv_path)
    T = readtable(csv_path);
    t = datetime(T.timestamp, 'ConvertFrom', 'posixtime');

    figure('Name','Per-frame EAR');
    plot(t, T.EAR, '-'); grid on;
    xlabel('Time'); ylabel('EAR');

    figure('Name','Per-frame MAR');
    plot(t, T.MAR, '-'); grid on;
    xlabel('Time'); ylabel('MAR');

    if any(strcmp('pitch_deg', T.Properties.VariableNames))
        figure('Name','Per-frame Pitch');
        plot(t, T.pitch_deg, '-'); grid on;
        xlabel('Time'); ylabel('Pitch (deg)');
    end
end
