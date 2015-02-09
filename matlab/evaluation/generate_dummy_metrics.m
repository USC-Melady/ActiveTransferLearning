function perf = generate_dummy_metrics(y)

perf = compute_fave_metrics(y, ones(size(y)), rand(size(y))/10+0.99);

end
