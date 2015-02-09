c1 = 5 + 2*sqrt(2);
% c1 = 1;
c2 = 5;
% c2 = 1;
C0 = [ 8 2 1 0.25 ];
alpha = 0.5;
k = 500;
m = [ 100 200 300 400 500 ];
col = { 'b', 'c', 'r', 'm', 'g' };
G = 0:0.01:1;

figure;
title('Varying m')
for i=1:length(C0)
    subplot(2, 2, i)
    title(sprintf('C0=%d', C0(i)))
    xlabel('G_k')
    ylabel('P_k')
    hold on;
    for j=1:length(m)
        P = zeros(size(G));
        for gi=1:length(G)
            epsk = C0(i) / (k-1);
            epsm = C0(i) / (2*m(j));
%             epsk = C0(i) / (k-1);
%             epsm = C0(i) / (2*m(j));
            Gbound = alpha * (sqrt(epsk) + epsk) + (1-alpha) * sqrt(epsm);
            fprintf('G=%.5f\t<= Gb=%.5f\n', G(gi), Gbound);
            if G(gi) <= Gbound
                P(gi) = 1;
            else
                temp = calculate_pk_tl(m(j), k, alpha, C0(i), c1, c2, G(gi));
                P(gi) = temp; %min(1, temp);
            end
        end
        plot(G, P, col{j});
%         keyboard
    end
end

k = [ 100 250 500 750 100 ];
m = 500;

c1 = 1;
c2 = 1;
figure;
title('Varying k')
for i=1:length(C0)
    subplot(2, 2, i)
    title(sprintf('C0=%d', C0(i)))
    xlabel('G_k')
    ylabel('P_k')
    hold on;
    for j=1:length(k)
        P = zeros(size(G));
        for gi=1:length(G)
            epsk = C0(i) / (k(j)-1);
            epsm = C0(i) / (2*m);
%             epsk = C0(i) / (k-1);
%             epsm = C0(i) / (2*m(j));
            Gbound = alpha * (sqrt(epsk) + epsk) + (1-alpha) * sqrt(epsm);
            fprintf('G=%.5f\t<= Gb=%.5f\n', G(gi), Gbound);
            if G(gi) <= Gbound
                P(gi) = 1;
            else
                temp = calculate_pk_tl(m, k(j), alpha, C0(i), c1, c2, G(gi));
                P(gi) = temp; %min(1, temp);
            end
        end
        plot(G, P, col{j});
%         keyboard
    end
end