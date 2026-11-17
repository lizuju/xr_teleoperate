import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from teleop.utils.camera_color import IMAGE_COLOR_PROFILE
from teleop.utils.act_contract import ACTION_DIM, CAMERAS, DEVICES, GROUPS, JOINT_NAMES, UNITS, validate_contract
from teleop.utils.act_dataset import action_chunk_indices, BalancedBatchSampler
from teleop.utils.act_runtime import split_action, join_action, PolicyGuard, prepare_training_image
from teleop.utils.icecream_task import IcecreamTask, phase_context, validate_annotation, source_digest


def metadata():
    return dict(state_dim=24, action_dim=24, joint_groups=GROUPS, end_effectors=DEVICES, units=UNITS,
                joint_names=JOINT_NAMES, camera_names=CAMERAS, image_transform_kind="letterbox_rgb_v1", image_color_profile=IMAGE_COLOR_PROFILE,
                gripper_position_limits={"right": {"min_q_rad": 4.28, "max_q_rad": 5.4}},
                controller_config={"dex1_slot": 33, "dex1_transport": "lowcmd", "dex1_kp": 5.,
                                   "dex1_kd": .05, "dex1_grip_delta": .5})


class ContractTests(unittest.TestCase):
    def test_all_layers_use_twenty_four_real_unit_values(self):
        self.assertEqual(len(JOINT_NAMES), ACTION_DIM)
        self.assertEqual(validate_contract(metadata()), (4.28, 5.4))
        values = np.arange(24.)
        parts = split_action(values)
        self.assertEqual(parts['right_q'].tolist(), [20.])
        self.assertEqual(parts['waist_q'], 21.)
        np.testing.assert_array_equal(join_action(*(parts[k] for k in ('arm_q','left_q','right_q','waist_q','head_q'))), values)

    def test_swapped_axes_and_old_gripper_units_are_refused(self):
        m = metadata(); m['joint_names'] = JOINT_NAMES[::-1]
        with self.assertRaises(ValueError): validate_contract(m)
        m = metadata(); m['units'] = ['normalized_0_to_1'] * 24
        with self.assertRaises(ValueError): validate_contract(m)

    def test_dex1_is_never_clipped_to_o6_normalized_range(self):
        stats = {k: [-2.] * 14 + [0.] * 6 + [4.28] + [-2.] * 3 for k in ['action_min','qpos_min']}
        stats.update({k: [2.] * 14 + [1.] * 6 + [5.4] + [2.] * 3 for k in ['action_max','qpos_max']})
        guard = PolicyGuard(stats, gripper_limits=(4.28,5.4))
        target = np.zeros(24); target[20] = 4.8
        self.assertEqual(guard.clamp(target)[20],4.8)
        target[20]=0; self.assertEqual(guard.clamp(target)[20],4.28)

    def test_letterbox_preserves_full_view_and_color(self):
        image = np.full((10,20,3),[10,20,30],dtype=np.uint8)
        actual = prepare_training_image(image,(20,20))
        np.testing.assert_array_equal(actual[10,10],[30,20,10])
        self.assertFalse(actual[:5].any()); self.assertFalse(actual[15:].any())


class TimingTests(unittest.TestCase):
    def test_resume_draws_same_batches_without_replaying_prefetch(self):
        dataset=SimpleNamespace(episodes=[{'anchors':4,'source_id':0},{'anchors':8,'source_id':1}])
        full=list(BalancedBatchSampler(dataset,3,0,10,42))
        resumed=list(BalancedBatchSampler(dataset,3,6,10,42))
        self.assertEqual(full[6:],resumed)

    def test_slow_recording_is_indexed_by_time_not_row_number(self):
        times = np.arange(10,dtype=np.int64)*30_000_000
        indices,pad=action_chunk_indices(times,0,6,40,int(times[-1]))
        self.assertEqual(indices.tolist(),[0,0,1,2,3,4])
        self.assertFalse(pad.any())

    def test_never_extrapolates_beyond_the_exported_continuous_segment(self):
        indices,pad=action_chunk_indices([0,25_000_000,50_000_000],0,8,40,50_000_000)
        self.assertEqual(pad.tolist(),[False,False,False,True,True,True,True,True])
        self.assertTrue((indices[3:]==2).all())

    def test_slow_hand_updates_retain_known_held_targets(self):
        indices,pad=action_chunk_indices([0,0,50_000_000,50_000_000,100_000_000],0,6,40,125_000_000)
        self.assertEqual(indices.tolist(),[0,1,3,3,4,4])
        self.assertFalse(pad.any())

    def test_backward_action_clock_is_refused(self):
        with self.assertRaises(ValueError):action_chunk_indices([100,90],0,3,40,120)


class TaskTests(unittest.TestCase):
    def test_manual_phases_hold_right_arm_gripper_waist_then_release(self):
        task=IcecreamTask((4.28,5.4));pose=np.arange(24.)/10
        task.start(0,pose);task.advance(1,pose);task.advance(2,pose)
        held=task.apply(np.zeros(24))
        np.testing.assert_array_equal(held[7:14],pose[7:14])
        np.testing.assert_array_equal(held[20:22],pose[20:22])
        self.assertTrue((held[:7]==0).all())
        task.advance(3,pose);task.advance(4,pose)
        self.assertFalse(task.apply(np.zeros(24)).any())
        task.advance(5,pose);self.assertTrue(task.done)

    def test_auto_requires_sustained_completion_and_feedback(self):
        task=IcecreamTask((4.28,5.4),auto=True);pose=np.zeros(24);pose[20]=5.4
        task.start(0,pose)
        self.assertFalse(task.observe(2,pose,1.))
        pose[20]=4.3
        self.assertFalse(task.observe(2.1,pose,1.))
        self.assertTrue(task.observe(2.5,pose,1.))
        self.assertEqual(task.phase,1)
        self.assertFalse(task.observe(4,pose,1.))

    def test_candidate_labels_are_not_silently_treated_as_reviewed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'frames.jsonl').write_text('{}\n')
            annotation={'reviewed':False,'frames_sha256':source_digest(root/'frames.jsonl'),
                        'boundaries':[0,2,4,6,8,10]}
            with self.assertRaises(ValueError):validate_annotation(annotation,root,10)
            annotation['reviewed']=True
            self.assertEqual(validate_annotation(annotation,root,10),[0,2,4,6,8,10])
            (root/'frames.jsonl').write_text('changed')
            with self.assertRaises(ValueError):validate_annotation(annotation,root,10)

    def test_correction_requires_human_success_episode(self):
        from record_r1_policy_result import record_result
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir();(run/'summary.json').write_text('{}')
            episode=root/'episode';episode.mkdir();(episode/'episode.json').write_text(json.dumps({'status':'complete','outcome':'discarded'}))
            with self.assertRaises(ValueError):record_result(run,'failure','grasp_cup','missed',[episode])
            (episode/'episode.json').write_text(json.dumps({'status':'complete','outcome':'success'}))
            doc=record_result(run,'intervention','grasp_cup','human correction',[episode])
            self.assertEqual(len(doc['corrections']),1)


class EvaluationTests(unittest.TestCase):
    def test_renamed_duplicate_source_cannot_be_independent(self):
        from eval_r1_act import evaluation_independence
        info = {"source_splits": {"train": [{"frames_sha256": "trained", "episode": "original"}],
                                   "val": [{"frames_sha256": "validated", "episode": "val"}]}}
        data = {"splits": {"test": [0]}, "episodes": [{"episode_id": 0, "source_id": 0}],
                "sources": [{"source_id": 0, "frames_sha256": "trained", "episode": "renamed"}]}
        self.assertFalse(evaluation_independence(info, data, "test"))
        data["sources"][0]["frames_sha256"] = "validated"
        self.assertFalse(evaluation_independence(info, data, "test"))
        data["sources"][0]["frames_sha256"] = "unseen"
        self.assertTrue(evaluation_independence(info, data, "test"))


class ConditionedModelTests(unittest.TestCase):
    def test_context_and_completion_survive_worker_checkpoint(self):
        import torch
        from teleop.utils.act_policy import ACTPolicy, save_checkpoint, load_policy, ChunkPredictor
        model=ACTPolicy(24,24,4,chunk_size=4,hidden_dim=32,nheads=2,dim_feedforward=32,
                        encoder_layers=1,vae_encoder_layers=1,dropout=0,pretrained=False,context_dim=8).eval()
        images=torch.zeros(1,4,3,48,64);qpos=torch.zeros(1,24)
        c1=torch.from_numpy(phase_context(0,0))[None]
        c2=torch.from_numpy(phase_context(2,3,(False,True)))[None]
        a=model(images,qpos,context=c1)
        b=model(images,qpos,context=c2)
        self.assertFalse(torch.allclose(a[0],b[0]))
        self.assertEqual(tuple(a[3].shape),(1,))
        training = model(images,qpos,torch.ones(1,4,24),torch.zeros(1,4,dtype=torch.bool),c1)
        torch.testing.assert_close(training[3],a[3])
        with self.assertRaises(ValueError):model(images,qpos)
        stats={k:([1.]*24 if k.endswith('std') else [0.]*24) for k in ['qpos_mean','qpos_std','action_mean','action_std']}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';save_checkpoint(path,model,stats,metadata(),1)
            restored,_=load_policy(path)
            prediction,complete=ChunkPredictor(restored,stats,'cpu')(
                np.zeros((4,48,64,3),np.uint8),np.zeros(24,np.float32),c2[0].numpy())
            self.assertEqual(prediction.shape,(4,24));self.assertTrue(0 <= complete <= 1)

    def test_reviewed_export_load_model_and_worker_protocol(self):
        import torch
        from test_training_export import TrainingExportTests
        from export_r1_training_dataset import export_dataset
        from teleop.utils.act_dataset import R1ACTDataset
        from teleop.utils.act_policy import ACTPolicy, act_loss
        from teleop.utils.act_runtime import encode_request, decode_request, encode_response, decode_response
        fixture = TrainingExportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        labels = {"episodes": {}}
        for number in range(3):
            directory, manifest, rows = fixture.episode(number=number, count=50, base_value=.1 + number * .1)
            info = manifest["info"]
            info.update(end_effectors=DEVICES, image_color_profile=IMAGE_COLOR_PROFILE, gripper_position_limits=metadata()["gripper_position_limits"],
                        launch_arguments=metadata()["controller_config"],
                        joint_names={group: [n.split("/")[1] for n in JOINT_NAMES if n.startswith(group + "/")]
                                     for group in GROUPS})
            for row in rows:
                row["states"]["right_ee"]["qpos"] = [4.9]
                row["actions"]["right_ee"]["qpos"] = [4.8]
                row["sample"]["aligned_states"]["hands"]["right"]["q"] = [4.9]
                row["sample"]["commands"]["hands"]["requested"]["right_q"] = [4.8]
                row["sample"]["commands"]["hands"]["published"]["right"]["q"] = [4.8]
            fixture.write(directory, manifest, rows)
            labels["episodes"][directory.name] = {"reviewed": True,
                "frames_sha256": source_digest(directory / "frames.jsonl"), "boundaries": [0,10,20,30,40,50]}
        label_path = fixture.root / "labels.json"
        label_path.write_text(json.dumps(labels))
        result = export_dataset(fixture.source, fixture.output, min_frames=2,
                                val_sources=["episode_0001"], test_sources=["episode_0002"],
                                image_hw=(48,64), annotations=label_path)
        validate_contract(result)
        dataset = R1ACTDataset(fixture.output, "train", chunk_size=4)
        images, qpos, target, pad, context, completion = dataset[8]
        self.assertEqual(pad.tolist(), [False,False,True,True])
        self.assertEqual(context[:5].tolist(), [1.,0.,0.,0.,0.])
        model = ACTPolicy(24,24,4,chunk_size=4,hidden_dim=32,nheads=2,dim_feedforward=32,
                          encoder_layers=1,vae_encoder_layers=1,dropout=0,pretrained=False,context_dim=8)
        pred, mu, logvar, completed = model(images[None],qpos[None],target[None],pad[None],context[None])
        loss, _, _ = act_loss(pred, target[None], pad[None], mu, logvar, 10.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        rgb = (images.permute(0,2,3,1).numpy() * 255).astype(np.uint8)
        q, image_copy, ctx = decode_request(encode_request(qpos.numpy(), rgb, context.numpy()),24,4,(48,64),8)
        np.testing.assert_array_equal(ctx, context.numpy())
        np.testing.assert_array_equal(image_copy, rgb)
        ms, actions, probability = decode_response(encode_response(12.,pred[0].detach().numpy(),.93),4,24)
        self.assertEqual(actions.shape,(4,24)); self.assertAlmostEqual(probability,.93)
        from teleop.utils.act_policy import save_checkpoint
        from train_r1_act import dataset_info
        from eval_r1_act import evaluate
        val = R1ACTDataset(fixture.output,"val",4)
        info = dataset_info(fixture.output,result,SimpleNamespace(dataset=dataset),SimpleNamespace(dataset=val))
        checkpoint = fixture.root / "conditioned.pt"
        save_checkpoint(checkpoint,model,result["normalization"],info,1)
        report = evaluate(checkpoint,split="val",batch_size=2,num_workers=0,device="cuda" if torch.cuda.is_available() else "cpu")
        self.assertTrue(report["independent"])
        self.assertEqual(set(report["phase_joint_mae"]),{'0','1','2','3','4'})
        self.assertEqual(report["groups"]["right_ee"]["unit"],"rad")

    def test_rng_restore_replays_random_sources(self):
        import random
        import torch
        from train_r1_act import rng_state, restore_rng
        before=rng_state()
        values=(random.random(),np.random.random(),torch.rand(3))
        restore_rng(before)
        self.assertEqual(random.random(),values[0]);self.assertEqual(np.random.random(),values[1])
        torch.testing.assert_close(torch.rand(3),values[2])


if __name__=='__main__':unittest.main()
