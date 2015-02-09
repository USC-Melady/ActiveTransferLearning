function [ y ] = map_labels(y, labels)

for i=1:length(y)
    if ~isnan(y(i))
        y(i) = find(labels == y(i));
    end
end
y(isnan(y)) = 0;

end
