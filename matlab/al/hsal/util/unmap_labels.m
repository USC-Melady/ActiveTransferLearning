function [ y ] = unmap_labels(y, labels)

for i=1:length(y)
    if y(i) > 0
        y(i) = labels(y(i));
    end
end
y(y==0) = nan;

end
