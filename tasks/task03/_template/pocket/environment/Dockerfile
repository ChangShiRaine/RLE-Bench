FROM rlebench-task03-agent:dev

COPY private/ /opt/private/
COPY public/ /
RUN chown -R root:root /opt/private /opt/rlebench \
    && chmod 700 /opt/private /var/lib/rlebench \
    && chmod -R a+rX /opt/rlebench \
    && chmod 755 /opt/*entrypoint.sh /opt/check_isolation.sh /opt/check_safety.sh \
    && chown -R agent:agent /workspace /home/agent

ENV RLEBENCH_IMAGE_LEVEL=L1 \
    RLEBENCH_LEVEL=L1 \
    RLEBENCH_TASK=RubikCube \
    RLEBENCH_MAX_STEPS_PER_TRIAL=@@INTERACTION_STEPS@@ \
    RLEBENCH_INTERACTION_STEPS=@@INTERACTION_STEPS@@ \
    RLEBENCH_EVALUATE_SECONDS=32400 \
    RLEBENCH_DEVELOP_SECONDS=32400
WORKDIR /workspace
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["sleep", "infinity"]
