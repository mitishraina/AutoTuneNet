from benchmarks.common.train_utils import train_epoch, evaluate
from autotunenet.logging.logger import get_logger
from autotunenet_bench.metrics.stability_metrics import StabilityMetrics
from autotunenet_bench.runners.logging_utils import BenchmarkLogger
from autotunenet_bench.metrics.compute_metrics import ComputeMetrics
import time

def train(
    model,
    optimizer,
    train_loader,
    val_loader,
    controller,
    regime,
    epochs,
    device,
    output_dir,
):
    model.to(device)

    logger = get_logger()
    train_dataset = train_loader.dataset

    stability = StabilityMetrics(reset_best_on_shift=True)
    compute = ComputeMetrics()
    bench_logger = BenchmarkLogger(output_dir)

    for epoch in range(epochs):
        was_active = regime._active if regime else False

        if regime is not None:
            regime.maybe_apply(
                epoch=epoch,
                train_dataset=train_dataset,
                optimizer=optimizer,
                model=model,
            )

        if regime and not was_active and regime._active:
            stability.mark_shift(epoch)
            
        # if regime and not was_active and regime._active:
        #     controller.on_regime_start(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)


        stability.record_loss(val_loss)
        stability.check_recovery(epoch, val_loss)

        start = time.perf_counter()

        controller.on_epoch_end(metric=-val_loss)
        controller.apply(optimizer)

        elapsed = time.perf_counter() - start
        compute.record_controller_time(elapsed)
        
        #compute metrics
        compute.step_epoch()
        if getattr(controller, "last_trial_start", False):
            compute.start_trial()
            
        if getattr(controller, "last_update", False):
            compute.record_update()
            compute.end_trial()
            
        if getattr(controller, "last_rollback", False):
            compute.record_rollback()

        #stability
        if controller.last_instability:
            stability.mark_instability(epoch)

        if controller.last_rollback:
            stability.mark_rollback(epoch)
            
            
        lr = float(optimizer.param_groups[0]["lr"])
        logger.info(
            f"Epoch {epoch+1} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={lr:.6f}"
        )

        bench_logger.log_epoch(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "hyperparams": {
                    k: v
                    for k, v in optimizer.param_groups[0].items()
                    if isinstance(v, (int, float))
                },
            }
        )

    bench_logger.write_epoch_logs()
    bench_logger.write_compute_metrics(compute.summary())
    bench_logger.write_stability_summary(stability.summary())
