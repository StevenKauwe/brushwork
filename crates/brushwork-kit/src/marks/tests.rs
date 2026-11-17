use super::*;

const INK: Ink = Ink { color: 0x3486ab, light: 0xffffff, dark: 0x101010 };
const PATH: [P; 5] = [[1.0, 2.0], [37.0, 6.0], [52.0, 41.0], [48.0, 75.0], [103.0, 50.0]];

fn stroke(seed: u64) -> StrokePlan {
    StrokePlan::new(&PATH, 24.0, INK, seed, StrokeOptions::default()).unwrap()
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a:?} != {b:?}");
}

fn valid(primitive: &Primitive) {
    let (color, alpha) = match *primitive {
        Primitive::Line { from, to, width, color, alpha } => {
            assert!(from.into_iter().chain(to).all(f64::is_finite));
            assert!(width.is_finite() && width > 0.0);
            (color, alpha)
        }
        Primitive::FillEllipse { center, radii, rotation, color, alpha } => {
            assert!(center.into_iter().all(f64::is_finite));
            assert!(radii.into_iter().all(|r| r.is_finite() && r > 0.0));
            assert!(rotation.is_finite());
            (color, alpha)
        }
        Primitive::StrokeEllipse { center, radii, rotation, width, color, alpha } => {
            assert!(center.into_iter().all(f64::is_finite));
            assert!(radii.into_iter().all(|r| r.is_finite() && r > 0.0));
            assert!(rotation.is_finite() && width.is_finite() && width > 0.0);
            (color, alpha)
        }
    };
    assert!(color <= 0x00ff_ffff);
    assert!(alpha.is_finite() && (0.0..=1.0).contains(&alpha));
    assert!(primitive.bounds().into_iter().all(f64::is_finite));
}

#[test]
fn random_stream_matches_mulberry32_javascript_including_zero_seed() {
    // Independently generated with Math.imul and unsigned shifts in Node.
    for (seed, expected) in [
        (0, [0.26642920868471265, 0.0003297457005828619, 0.2232720274478197, 0.1462021479383111, 0.46732782293111086, 0.5450490827206522]),
        (1, [0.6270739405881613, 0.002735721180215478, 0.5274470399599522, 0.9810509674716741, 0.9683778982143849, 0.281103502959013]),
        (0x12345678, [0.10615200875326991, 0.941276284167543, 0.9398706152569503, 0.2338848018553108, 0.9045877147000283, 0.778330324916169]),
        (0xffffffff, [0.8964226141106337, 0.189478256739676, 0.7156526781618595, 0.9440599093213677, 0.8452364315744489, 0.5391399988438934]),
    ] {
        let mut rng = Rng::new(seed);
        assert_eq!(std::array::from_fn::<_, 6, _>(|_| rng.next()), expected);
    }
    assert_ne!(Rng::new(5).next(), Rng::new(5 + (1 << 32)).next());
}

#[test]
fn pressure_profiles_match_original_numeric_defaults() {
    let progress = [0.0, 0.04, 0.25, 0.5, 0.75, 1.0];
    let lift = [0.4834184099682108, 0.4834184099682108, 0.8857675191023606, 1.0, 0.8857675191023606, 0.000002696520945740995];
    let flick = [0.12, 0.1870086186427989, 0.41029174037003674, 0.6254272761986953, 0.819087735092216, 1.0];
    for i in 0..progress.len() {
        close(PressureProfile::Lift.at(progress[i]).unwrap(), lift[i]);
        close(PressureProfile::Flick.at(progress[i]).unwrap(), flick[i]);
    }
    assert_eq!(StrokeOptions::default(), StrokeOptions { body: 0.9, pressure: PressureProfile::Lift, dry: 0.9, step: 3.0 });
    for profile in [PressureProfile::Lift, PressureProfile::Flick] {
        assert_eq!(profile.at(-10.0), profile.at(0.0));
        assert_eq!(profile.at(10.0), profile.at(1.0));
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(profile.at(bad).is_err());
        }
    }
}

#[test]
fn resampling_carries_distance_through_corners_and_keeps_exact_endpoints() {
    assert_eq!(resample(&[[2.0, 3.0], [6.0, 3.0], [6.0, 7.0]], 3.0).unwrap(), [[2.0, 3.0], [5.0, 3.0], [6.0, 5.0], [6.0, 7.0]]);
    assert_eq!(resample(&[[0.0, 0.0], [9.0, 0.0]], 3.0).unwrap(), [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]]);
    assert_eq!(resample(&[[0.0, 0.0], [9.0, 0.0]], 100.0).unwrap(), [[0.0, 0.0], [9.0, 0.0]]);
    let result = resample(&PATH, 3.0).unwrap();
    assert_eq!(result.first(), PATH.first());
    assert_eq!(result.last(), PATH.last());
    assert!(result.windows(2).all(|w| w[0] != w[1]));
}

#[test]
fn duplicate_vertices_do_not_change_resampling_or_the_stroke() {
    let mut duplicate = Vec::new();
    for point in PATH {
        duplicate.extend([point, point, point]);
    }
    assert_eq!(resample(&PATH, 3.0), resample(&duplicate, 3.0));
    let mut a = stroke(71);
    let mut b = StrokePlan::new(&duplicate, 24.0, INK, 71, StrokeOptions::default()).unwrap();
    assert_eq!(a.advance(1.0), b.advance(1.0));
}

#[test]
fn empty_singleton_and_coincident_paths_are_valid_and_emit_nothing() {
    for points in [vec![], vec![[2.0, 3.0]], vec![[2.0, 3.0]; 16]] {
        let path = resample(&points, 3.0).unwrap();
        assert_eq!(path.len(), usize::from(!points.is_empty()));
        let mut plan = StrokePlan::new(&points, 20.0, INK, 1, StrokeOptions::default()).unwrap();
        assert_eq!(plan.segment_count(), 0);
        for progress in [0.0, 0.5, 1.0, 2.0] {
            let commands = plan.advance(progress).unwrap();
            assert!(commands.is_empty());
            assert_eq!(commands.capacity(), 0);
        }
    }
}

#[test]
fn closed_path_reaches_its_start_again_without_duplicate_adjacent_vertices() {
    let points = [[0.0, 0.0], [9.0, 0.0], [9.0, 12.0], [0.0, 0.0]];
    let mut plan = StrokePlan::new(&points, 10.0, INK, 3, StrokeOptions::default()).unwrap();
    assert_eq!(plan.path().first(), plan.path().last());
    assert!(plan.path().windows(2).all(|w| w[0] != w[1]));
    assert!(!plan.advance(1.0).unwrap().is_empty());
    assert_eq!(plan.drawn_segments(), plan.segment_count());
}

#[test]
fn progress_is_monotone_and_failed_progress_is_transactional() {
    let mut plan = stroke(8);
    assert_eq!(plan.drawn_segments(), 0);
    assert!(!plan.advance(0.5).unwrap().is_empty());
    let saved = plan.clone();
    for progress in [0.5, 0.25, 0.0, -1.0] {
        let commands = plan.advance(progress).unwrap();
        assert_eq!(commands.capacity(), 0);
        assert_eq!(plan, saved);
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(plan.advance(bad).is_err());
        assert_eq!(plan, saved);
    }
    assert!(!plan.advance(2.0).unwrap().is_empty());
    assert_eq!(plan.drawn_segments(), plan.segment_count());
    assert!(plan.advance(1.0).unwrap().is_empty());
}

#[test]
fn drawing_in_any_chunks_equals_all_at_once_byte_for_byte() {
    for seed in 0..24 {
        for pressure in [PressureProfile::Lift, PressureProfile::Flick] {
            let options = StrokeOptions { pressure, ..StrokeOptions::default() };
            let plan = StrokePlan::new(&PATH, 24.0, INK, seed, options).unwrap();
            let all = plan.clone().advance(1.0).unwrap();
            for steps in [1, 2, 7, 60, 137] {
                let mut partial = plan.clone();
                let mut commands = Vec::new();
                for i in 0..=steps {
                    commands.extend(partial.advance(i as f64 / steps as f64).unwrap());
                }
                assert_eq!(commands, all, "seed {seed}, {pressure:?}, {steps} steps");
                assert_eq!(partial.drawn_segments(), partial.segment_count());
            }
        }
    }
}

#[test]
fn progress_chunks_can_skip_and_repeat_frames_without_repainting() {
    let mut a = stroke(91);
    let mut commands = Vec::new();
    for p in [0.0, 0.1, 0.1, 0.35, 0.2, 0.9, 0.9, 1.0, 1.0] {
        commands.extend(a.advance(p).unwrap());
    }
    assert_eq!(commands, stroke(91).advance(1.0).unwrap());
}

#[test]
fn seed_is_reproducible_and_independent_of_other_marks_or_creation_order() {
    let expected = stroke(0x12345678).advance(1.0).unwrap();
    let mut unrelated = stroke(0xabcdef);
    unrelated.advance(0.7).unwrap();
    for kind in [StampKind::Dab, StampKind::Flick, StampKind::Spatter] {
        Stamp::new(kind, 22.0, INK, 88).unwrap();
    }
    assert_eq!(stroke(0x12345678).advance(1.0).unwrap(), expected);
    assert_ne!(stroke(0x12345679).advance(1.0).unwrap(), expected);
    assert_ne!(stroke(0x112345678).advance(1.0).unwrap(), expected);
    let a = Stamp::new(StampKind::Flick, 24.0, INK, 5).unwrap();
    let b = Stamp::new(StampKind::Spatter, 24.0, INK, 8).unwrap();
    assert_eq!(b, Stamp::new(StampKind::Spatter, 24.0, INK, 8).unwrap());
    assert_eq!(a, Stamp::new(StampKind::Flick, 24.0, INK, 5).unwrap());
    assert_ne!(a, Stamp::new(StampKind::Flick, 24.0, INK, 6).unwrap());
}

#[test]
fn brush_colour_has_no_implicit_palette() {
    let black = Ink { color: 0, light: 0, dark: 0 };
    let white = Ink { color: 0xffffff, light: 0xffffff, dark: 0xffffff };
    for (ink, expected) in [(black, 0), (white, 0xffffff)] {
        let commands = StrokePlan::new(&PATH, 24.0, ink, 17, StrokeOptions::default()).unwrap().advance(1.0).unwrap();
        assert!(commands.iter().all(|p| matches!(p, Primitive::Line { color, .. } if *color == expected)));
        for primitive in Stamp::new(StampKind::Spatter, 24.0, ink, 17).unwrap().primitives {
            match primitive {
                Primitive::FillEllipse { color, .. } | Primitive::StrokeEllipse { color, .. } => assert_eq!(color, expected),
                _ => panic!("unexpected line in spatter"),
            }
        }
    }
    assert_eq!(INK.shade(0.0), INK.color);
    assert_eq!(INK.shade(1.0), INK.light);
    assert_eq!(INK.shade(-1.0), INK.dark);
}

#[test]
fn a_strokes_style_does_not_depend_on_canvas_position() {
    let points = [[0.0, 0.0], [63.0, 0.0]];
    let moved = [[64.0, 32.0], [127.0, 32.0]];
    let a = StrokePlan::new(&points, 12.0, INK, 91, StrokeOptions::default()).unwrap();
    let b = StrokePlan::new(&moved, 12.0, INK, 91, StrokeOptions::default()).unwrap();
    assert_eq!(a.bristles, b.bristles);
    assert_eq!(a.swell, b.swell);
    let mut expected = a.clone().advance(1.0).unwrap();
    for primitive in &mut expected {
        primitive.translate([64.0, 32.0]);
    }
    assert_eq!(expected, b.clone().advance(1.0).unwrap());
}

#[test]
fn incremental_dab_uses_seeded_geometry_without_timing() {
    let center = [300.0, 120.0];
    let mut dab = StrokePlan::dab(center, 20.0, INK, 77).unwrap();
    let first = dab.path()[0];
    let last = *dab.path().last().unwrap();
    close((first[0] + last[0]) / 2.0, center[0]);
    close((first[1] + last[1]) / 2.0, center[1]);
    assert!((10.0..18.0).contains(&(last[0] - first[0]).hypot(last[1] - first[1])));
    let mut commands = dab.advance(0.5).unwrap();
    commands.extend(dab.advance(1.0).unwrap());
    assert_eq!(commands, StrokePlan::dab(center, 20.0, INK, 77).unwrap().advance(1.0).unwrap());
    assert_ne!(commands, StrokePlan::dab(center, 20.0, INK, 78).unwrap().advance(1.0).unwrap());
}

#[test]
fn spatter_is_five_filled_drops_with_lighter_rims_in_paint_order() {
    let stamp = Stamp::new(StampKind::Spatter, 20.0, INK, 42).unwrap();
    assert_eq!(stamp.primitives.len(), 10);
    for pair in stamp.primitives.as_chunks::<2>().0 {
        let Primitive::FillEllipse { center, radii, rotation, alpha, .. } = pair[0] else { panic!("fill must precede rim") };
        assert_eq!(alpha, 1.0);
        assert_eq!(pair[1], Primitive::StrokeEllipse { center, radii, rotation, width: (radii[1] * 0.25).max(0.8), color: INK.shade(0.35), alpha: 0.45 });
    }
}

#[test]
fn stamp_bounds_include_round_caps_rotated_ellipses_and_rims() {
    for seed in 0..96 {
        for size in [0.001, 0.1, 0.5, 1.0, 3.0, 8.0, 12.0, 22.0, 34.0, 128.0] {
            for kind in [StampKind::Dab, StampKind::Flick, StampKind::Spatter] {
                let stamp = Stamp::new(kind, size, INK, seed).unwrap();
                assert!(stamp.width > 0.0 && stamp.height > 0.0);
                assert_eq!(stamp.width.fract(), 0.0);
                assert_eq!(stamp.height.fract(), 0.0);
                for primitive in &stamp.primitives {
                    valid(primitive);
                    let [left, top, right, bottom] = primitive.bounds();
                    assert!(left >= 0.0 && top >= 0.0 && right <= stamp.width && bottom <= stamp.height, "{kind:?}, size {size}, seed {seed}: {primitive:?}");
                }
            }
        }
    }
}

#[test]
fn normal_stamps_keep_original_dimensions_and_do_not_need_extra_padding() {
    for seed in 0..128 {
        for size in [8.0, 10.0, 12.0, 17.3, 20.0, 22.0, 24.0, 34.0] {
            for kind in [StampKind::Dab, StampKind::Flick, StampKind::Spatter] {
                let stamp = Stamp::new(kind, size, INK, seed).unwrap();
                let len = size
                    * match kind {
                        StampKind::Dab => 0.8,
                        StampKind::Flick => 2.6,
                        StampKind::Spatter => 1.0,
                    };
                let pad = size * 0.7;
                assert_eq!(stamp.width, (len + pad * 2.0).ceil(), "{kind:?}, size {size}, seed {seed}");
                assert_eq!(stamp.height, (size * 1.4 + pad * 2.0).ceil(), "{kind:?}, size {size}, seed {seed}");
            }
        }
    }
}

#[test]
fn geometry_bounds_are_exact_for_axis_aligned_examples() {
    assert_eq!(Primitive::Line { from: [2.0, 3.0], to: [8.0, 9.0], width: 4.0, color: 0, alpha: 1.0 }.bounds(), [0.0, 1.0, 10.0, 11.0]);
    assert_eq!(Primitive::FillEllipse { center: [5.0, 5.0], radii: [2.0, 3.0], rotation: 0.0, color: 0, alpha: 1.0 }.bounds(), [3.0, 2.0, 7.0, 8.0]);
    assert_eq!(
        Primitive::StrokeEllipse { center: [5.0, 5.0], radii: [2.0, 3.0], rotation: 0.0, width: 2.0, color: 0, alpha: 1.0 }.bounds(),
        [2.0, 1.0, 8.0, 9.0]
    );
}

#[test]
fn invalid_dimensions_options_points_and_colours_are_rejected() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(resample(&PATH, bad).is_err());
        assert!(StrokePlan::new(&PATH, bad, INK, 0, StrokeOptions::default()).is_err());
        assert!(StrokePlan::dab([0.0, 0.0], bad, INK, 0).is_err());
        for kind in [StampKind::Dab, StampKind::Flick, StampKind::Spatter] {
            assert!(Stamp::new(kind, bad, INK, 0).is_err());
        }
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for point in [[bad, 0.0], [0.0, bad]] {
            assert!(resample(&[point], 3.0).is_err());
            assert!(StrokePlan::dab(point, 24.0, INK, 0).is_err());
        }
    }
    for options in [
        StrokeOptions { body: -0.1, ..StrokeOptions::default() },
        StrokeOptions { body: 1.1, ..StrokeOptions::default() },
        StrokeOptions { body: f64::NAN, ..StrokeOptions::default() },
        StrokeOptions { dry: -0.1, ..StrokeOptions::default() },
        StrokeOptions { dry: f64::INFINITY, ..StrokeOptions::default() },
        StrokeOptions { step: f64::NAN, ..StrokeOptions::default() },
    ] {
        assert!(StrokePlan::new(&PATH, 24.0, INK, 0, options).is_err());
    }
    for bad in [Ink { color: 0x1000000, ..INK }, Ink { light: u32::MAX, ..INK }, Ink { dark: 0x1000000, ..INK }] {
        assert!(StrokePlan::new(&PATH, 24.0, bad, 0, StrokeOptions::default()).is_err());
        assert!(Stamp::new(StampKind::Spatter, 24.0, bad, 0).is_err());
    }
}

#[test]
fn untrusted_work_sizes_are_bounded_before_commands_are_allocated() {
    assert!(matches!(resample(&vec![[0.0, 0.0]; MAX_PATH_POINTS + 1], 3.0), Err(MarkError::LimitExceeded(_))));
    for step in [f64::from_bits(1), f64::MIN_POSITIVE, 0.000001] {
        assert!(matches!(resample(&[[0.0, 0.0], [100.0, 0.0]], step), Err(MarkError::LimitExceeded(_))));
    }
    assert!(matches!(resample(&[[MAX_COORDINATE + 1.0, 0.0]], 1.0), Err(MarkError::LimitExceeded(_))));
    assert!(matches!(StrokePlan::new(&PATH, MAX_WIDTH + 1.0, INK, 0, StrokeOptions::default()), Err(MarkError::LimitExceeded(_))));
    assert!(matches!(Stamp::new(StampKind::Dab, MAX_STAMP_SIZE + 1.0, INK, 0), Err(MarkError::LimitExceeded(_))));
    assert!(matches!(
        StrokePlan::new(&[[0.0, 0.0], [10_000.0, 0.0]], 40.0, INK, 0, StrokeOptions { step: 1.0, ..StrokeOptions::default() }),
        Err(MarkError::LimitExceeded("stroke primitives"))
    ));
    assert_eq!(resample(&[[1.0, 2.0]; 20], f64::from_bits(1)).unwrap(), [[1.0, 2.0]]);
}

#[test]
fn exact_resampling_capacity_is_accepted_and_the_next_sample_is_rejected() {
    let last = [(MAX_RESAMPLED_POINTS - 1) as f64, 0.0];
    let points = resample(&[[0.0, 0.0], last], 1.0).unwrap();
    assert_eq!(points.len(), MAX_RESAMPLED_POINTS);
    assert_eq!(points.last(), Some(&last));
    assert!(matches!(resample(&[[0.0, 0.0], [last[0] + 0.01, 0.0]], 1.0), Err(MarkError::LimitExceeded(_))));
}

#[test]
fn extremes_never_emit_nonfinite_or_invalid_commands() {
    for size in [f64::from_bits(1), f64::MIN_POSITIVE, 1e-200, 0.001, MAX_STAMP_SIZE] {
        for kind in [StampKind::Dab, StampKind::Flick, StampKind::Spatter] {
            let stamp = Stamp::new(kind, size, INK, 7).unwrap();
            for primitive in &stamp.primitives {
                valid(primitive);
            }
        }
    }
    for width in [f64::from_bits(1), f64::MIN_POSITIVE, 0.001, MAX_WIDTH] {
        let commands =
            StrokePlan::new(&[[MAX_COORDINATE - 5.0, 0.0], [MAX_COORDINATE, 0.0]], width, INK, 9, StrokeOptions::default()).unwrap().advance(1.0).unwrap();
        for primitive in &commands {
            valid(primitive);
        }
    }
}
