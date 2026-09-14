FROM rlebench-task03-agent:dev

COPY private/harness/tabletop/budgets.py /opt/private/harness/tabletop/budgets.py
COPY private/rlebench /opt/private/rlebench
COPY private/harness/tabletop/hidden_com /opt/private/harness/tabletop/hidden_com
RUN rm -f /opt/HARNESS_MANUAL.md \
    && rm -rf /opt/rlebench/harness \
    && mkdir -p /opt/rlebench/harness \
    && touch /opt/rlebench/harness/__init__.py
COPY public/client.py /opt/rlebench/harness/client.py
COPY entrypoint.sh /opt/hidden_com_entrypoint.sh
RUN chmod 700 /opt/private /var/lib/rlebench \
    && chown -R root:root /opt/private /var/lib/rlebench /opt/rlebench \
    && chmod -R a+rX /opt/rlebench \
    && chmod 755 /opt/hidden_com_entrypoint.sh \
    && /usr/local/bin/python -I -c 'import mujoco; assert mujoco.__version__ == "3.3.1"'

WORKDIR /workspace
ENTRYPOINT ["/opt/hidden_com_entrypoint.sh"]
CMD ["sleep", "infinity"]
