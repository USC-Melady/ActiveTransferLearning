function [ D ] = compute_tree_depth(tree)

D = [];
P = tree.root;
Dp = 0;
while ~isempty(P)
    v = P(1); P(1) = [];
    d = Dp(1); Dp(1) = [];
    if v > 0
        P = [ P tree.children(v,:) ];
        Dp = [ Dp d+1 d+1 ];
        D = [ D; v d ];
    end
end

end

